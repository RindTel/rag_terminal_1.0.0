"""
eval/run_eval.py
────────────────
Baseline evaluation harness for the qwen-rag pipeline.

Runs every question in golden_set.json through the REAL pipeline
(DocumentProcessor → VectorStore → OllamaClient) and scores:

  RETRIEVAL   doc_hit        — did we retrieve from the expected document?
              context_recall — what fraction of the required facts made it
                               into the retrieved context?
              retrieval_hit  — did ALL required facts make it in?

  ANSWER      fact_recall    — fraction of required facts present in the answer
              abstained      — did the model correctly refuse when it should?
              judge          — optional LLM-as-judge (Ollama), CORRECT/PARTIAL/INCORRECT

  DIAGNOSTIC  prompt_tokens  — EXACT Qwen prompt token count, read from Ollama's
                               prompt_eval_count. Compared against the real usable
                               budget (num_ctx - num_predict) to detect silent
                               context-window overflow.

This script FIXES NOTHING. It exercises the pipeline exactly as it ships,
against an isolated index (eval/.index) so the user's real vectorstore/ is
never touched.

Usage:
    python eval/run_eval.py                 # full run, with LLM judge
    python eval/run_eval.py --no-judge      # deterministic scoring only
    python eval/run_eval.py --top-k 3       # override retrieval depth
    python eval/run_eval.py --retrieval-only  # skip generation (no Ollama needed)
"""

import os
import sys
import json
import time
import argparse
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from src.rag_pipeline import RAGPipeline, NO_RELEVANT_CONTEXT_MESSAGE
from src.llm_client import SYSTEM_PROMPT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_DIR = os.path.join(HERE, "corpus")
INDEX_DIR = os.path.join(HERE, ".index")
GOLDEN = os.path.join(HERE, "golden_set.json")

ABSTAIN_MARKERS = [
    "couldn't find", "could not find", "cannot find", "can't find",
    "does not contain", "doesn't contain", "no information",
    "not mentioned", "not provided", "not specified", "not found in",
    "unable to answer", "no clear answer", "not enough information",
]


# ── matching ────────────────────────────────────────────────────────────────

def norm(s: str) -> str:
    """Lowercase + collapse whitespace. Keeps punctuation so 'P2002' and
    'localhost:4000/api/v1' still match literally."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def facts_found(facts, text: str):
    """Return the subset of key_facts literally present in text."""
    t = norm(text)
    return [f for f in facts if norm(f) in t]


def recall(facts, text: str) -> float:
    if not facts:
        return 1.0
    return len(facts_found(facts, text)) / len(facts)


def abstained(answer: str) -> bool:
    a = norm(answer)
    return any(m in a for m in ABSTAIN_MARKERS)


# ── generation (faithful to OllamaClient, but reads token stats) ────────────

def generate_with_stats(llm, prompt: str, seed: int = 0):
    """
    Same request OllamaClient.generate() makes — same model, same system prompt,
    same options, pulled off the live client object so it cannot drift.

    Two deliberate differences, both for measurement only:
      - reads prompt_eval_count (exact prompt tokens Qwen consumed), which
        OllamaClient throws away;
      - pins `seed` so runs are reproducible. Temperature is pinned by the
        caller (see --temperature); the app's own runtime default is untouched.
    """
    payload = {
        "model": llm.model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "num_ctx": llm.num_ctx,
            "num_predict": llm.num_predict,
            "seed": seed,
        },
    }
    t0 = time.time()
    r = requests.post(f"{llm.base_url}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    d = r.json()
    return {
        "answer": d.get("response", "").strip(),
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "gen_tokens": d.get("eval_count", 0),
        "latency": time.time() - t0,
    }


JUDGE_SYSTEM = (
    "You are a strict grader for a retrieval-augmented QA system. "
    "You compare a candidate answer against a reference answer and required facts. "
    "Reply with ONLY a JSON object, no prose."
)

JUDGE_TEMPLATE = """QUESTION:
{q}

REFERENCE ANSWER (ground truth):
{ref}

REQUIRED FACTS the answer must convey:
{facts}

CANDIDATE ANSWER:
{cand}

Grade the candidate. Ignore wording, style, and extra detail — grade only factual correctness.
- "CORRECT"   = conveys all required facts, states nothing false
- "PARTIAL"   = some required facts present or missing, nothing false
- "INCORRECT" = misses the point, or asserts something false/hallucinated

Reply with ONLY: {{"verdict": "CORRECT|PARTIAL|INCORRECT", "reason": "<10 words"}}"""


def judge(llm, q, ref, facts, cand):
    prompt = JUDGE_TEMPLATE.format(
        q=q, ref=ref, facts="\n".join(f"- {f}" for f in facts) or "(none — must abstain)", cand=cand
    )
    payload = {
        "model": llm.model,
        "prompt": prompt,
        "system": JUDGE_SYSTEM,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 120},
    }
    try:
        r = requests.post(f"{llm.base_url}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        raw = r.json().get("response", "")
        m = re.search(r'\{.*\}', raw, re.S)
        if not m:
            return "UNPARSED", raw[:40]
        obj = json.loads(m.group(0))
        v = str(obj.get("verdict", "UNPARSED")).upper()
        return (v if v in ("CORRECT", "PARTIAL", "INCORRECT") else "UNPARSED"), obj.get("reason", "")
    except Exception as e:
        return "ERROR", str(e)[:40]


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=None, help="override retrieval depth")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge")
    ap.add_argument("--retrieval-only", action="store_true", help="skip generation entirely")
    ap.add_argument("--model", default=None, help="override Ollama model")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="EVAL-ONLY generation temperature. Pinned to 0 so runs are "
                         "reproducible and a 1-query delta means something. The app's "
                         "own default (LLM_TEMPERATURE=0.1) is unaffected.")
    ap.add_argument("--seed", type=int, default=0, help="eval-only sampling seed")
    ap.add_argument("--allow-empty-context", action="store_true",
                    help="Reproduce PRE-FIX behaviour: when retrieval returns nothing, "
                         "still call the LLM with an empty CONTEXT block. Use to A/B the "
                         "grounding guard without reverting src/.")
    ap.add_argument("--no-hybrid", action="store_true",
                    help="Force the pure-vector retrieval path (overrides "
                         "config.HYBRID_RETRIEVAL). Use to A/B hybrid vs vector-only.")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="Disable conversational query rewriting (overrides "
                         "config.QUERY_REWRITE). Use to A/B follow-up rewriting.")
    ap.add_argument("--out", default=os.path.join(HERE, "report.md"))
    ap.add_argument("--label", default="baseline", help="label for this run")
    args = ap.parse_args()

    # A/B overrides — flip retrieval behaviour for this run without editing config.
    import config
    if args.no_hybrid:
        config.HYBRID_RETRIEVAL = False
    if args.no_rewrite:
        config.QUERY_REWRITE = False

    gold = json.load(open(GOLDEN))
    questions = gold["questions"]

    # Isolated pipeline — never touches the user's uploads/ or vectorstore/.
    pipe = RAGPipeline(uploads_dir=CORPUS_DIR, vectorstore_dir=INDEX_DIR)
    if args.model:
        pipe.llm.model = args.model
    if args.top_k:
        pipe.top_k = args.top_k
    pipe.llm.temperature = args.temperature

    print(f"\n{'='*78}\n  RAG EVAL — {args.label}\n{'='*78}")
    print(f"  corpus     : {CORPUS_DIR}")
    print(f"  index      : {INDEX_DIR}  (isolated)")
    print(f"  embedding  : {pipe.vector_store.model_name}")
    print(f"  llm        : {pipe.llm.model}  (temp={pipe.llm.temperature}, seed={args.seed} — pinned for reproducibility)")
    if args.allow_empty_context:
        print(f"  MODE       : --allow-empty-context  (PRE-FIX behaviour: LLM called with empty context)")
    import config as _cfg
    if _cfg.HYBRID_RETRIEVAL:
        print(f"  retrieval  : HYBRID (BM25 + vector, RRF fused"
              f"{', reranked' if _cfg.RERANK else ''})"
              f"  rerank_gate={_cfg.RERANK_SCORE_THRESHOLD}  candidates={_cfg.HYBRID_CANDIDATES}")
    else:
        print(f"  retrieval  : VECTOR-ONLY  score_threshold={pipe.score_threshold}")
    print(f"  query rewrite: {'ON' if _cfg.QUERY_REWRITE else 'OFF'}")
    print(f"  top_k      : {pipe.top_k}")
    print(f"  chunking   : {pipe.processor.chunk_tokens}tok / "
          f"{pipe.processor.chunk_overlap_tokens}tok overlap "
          f"(encoder limit {pipe.vector_store.get_max_seq_length()})")
    budget = pipe.llm.num_ctx - pipe.llm.num_predict
    print(f"  num_ctx    : {pipe.llm.num_ctx}  num_predict: {pipe.llm.num_predict}"
          f"  -> usable prompt budget: {budget} tokens")

    # Fresh, deterministic index: clear then ingest each file exactly once.
    print("\n  Building index from corpus ...")
    pipe.vector_store.clear()
    files = [os.path.join(CORPUS_DIR, f) for f in sorted(os.listdir(CORPUS_DIR))
             if f.lower().endswith((".pdf", ".txt"))]
    added, errs = pipe.ingest_files(files)
    for e in errs:
        print("   !", e)
    print(f"  Indexed {added} chunks from {len(files)} file(s): "
          f"{', '.join(os.path.basename(f) for f in files)}\n")

    do_gen = not args.retrieval_only
    do_judge = do_gen and not args.no_judge
    if do_gen and not pipe.llm.is_available():
        print("  !! Ollama not reachable — falling back to --retrieval-only\n")
        do_gen = do_judge = False

    rows = []
    for i, q in enumerate(questions, 1):
        facts = q.get("key_facts", [])
        exp_docs = set(q.get("expected_source_docs", []))
        hist = q.get("chat_history")
        unanswerable = q["category"] == "unanswerable"

        t0 = time.time()
        hits = pipe.retrieve(q["query"], chat_history=hist)
        r_lat = time.time() - t0

        chunks = [c for c, _ in hits]
        scores = [s for _, s in hits]
        got_docs = {c["source_file"] for c in chunks}
        context = "\n\n".join(c["text"] for c in chunks)

        ctx_recall = recall(facts, context)
        doc_hit = exp_docs.issubset(got_docs) if exp_docs else (len(got_docs) == 0 or True)
        retr_hit = (ctx_recall >= 1.0) and doc_hit

        row = {
            "id": q["id"], "category": q["category"], "query": q["query"],
            "n_retrieved": len(chunks),
            "top_score": round(max(scores), 3) if scores else None,
            "docs": sorted(got_docs),
            "doc_hit": doc_hit,
            "context_recall": ctx_recall,
            "retrieval_hit": retr_hit,
            "missing_from_context": [f for f in facts if f not in facts_found(facts, context)],
            "retrieval_latency": r_lat,
        }

        row["no_retrieval"] = (len(chunks) == 0)

        if do_gen and row["no_retrieval"] and args.allow_empty_context:
            # PRE-FIX behaviour, for A/B only: hand the LLM an empty CONTEXT block
            # and let it answer from parametric memory.
            prompt = pipe.llm.build_rag_prompt(q["query"], [], chat_history=hist)
            g = generate_with_stats(pipe.llm, prompt, seed=args.seed)
            ans = g["answer"]
            row.update({
                "answer": ans, "generated": True,
                "prompt_tokens": g["prompt_tokens"],
                "overflow": max(0, g["prompt_tokens"] - budget),
                "gen_tokens": g["gen_tokens"], "gen_latency": g["latency"],
                "fact_recall": recall(facts, ans), "abstained": abstained(ans),
            })
            row["pass"] = row["abstained"] if unanswerable else (row["fact_recall"] >= 1.0)
            if do_judge:
                v, why = judge(pipe.llm, q["query"], q["expected_answer"], facts, ans)
                row["judge"], row["judge_reason"] = v, why
                if not unanswerable:
                    row["pass"] = (v == "CORRECT")

        elif do_gen and row["no_retrieval"]:
            # The pipeline now refuses without calling the LLM. Mirror that here:
            # no generation, no judging — this is NOT a grounded answer and must
            # not be scored as one.
            row.update({
                "answer": NO_RELEVANT_CONTEXT_MESSAGE,
                "generated": False,
                "prompt_tokens": 0, "overflow": 0, "gen_tokens": 0, "gen_latency": 0.0,
                "fact_recall": 0.0,
                "abstained": True,
                "judge": "NO_RETR",
                "judge_reason": "refused without calling LLM",
            })
            # Correct behaviour on an unanswerable question; a retrieval failure
            # on anything else. Either way, never a hallucination.
            row["pass"] = unanswerable

        elif do_gen:
            build = pipe.llm.assemble(q["query"], chunks, chat_history=hist)
            try:
                g = generate_with_stats(pipe.llm, build.prompt, seed=args.seed)
            except Exception as e:
                g = {"answer": f"[GENERATION ERROR: {e}]", "prompt_tokens": 0,
                     "gen_tokens": 0, "latency": 0.0}
            ans = g["answer"]
            row.update({
                "answer": ans,
                "generated": True,
                "prompt_tokens": g["prompt_tokens"],
                "overflow": max(0, g["prompt_tokens"] - budget),
                "gen_tokens": g["gen_tokens"],
                "gen_latency": g["latency"],
                "fact_recall": recall(facts, ans),
                "abstained": abstained(ans),
                # Budgeter accounting
                "chunks_used": len(build.used_chunks),
                "chunks_dropped": len(build.dropped_chunks),
                "est_tokens": build.est_tokens,
                # THE load-bearing safety property: our estimate must never come in
                # UNDER Ollama's real count. If it does, the budgeter is handing the
                # model more than it can hold and we are back to silent truncation.
                "est_undercount": (
                    g["prompt_tokens"] > 0 and build.est_tokens < g["prompt_tokens"]
                ),
            })
            if unanswerable:
                row["pass"] = row["abstained"]
            else:
                row["pass"] = row["fact_recall"] >= 1.0

            if do_judge:
                v, why = judge(pipe.llm, q["query"], q["expected_answer"], facts, ans)
                row["judge"] = v
                row["judge_reason"] = why
                if unanswerable:
                    row["pass"] = row["abstained"]
                else:
                    row["pass"] = (v == "CORRECT")
        else:
            row["pass"] = retr_hit

        rows.append(row)

        # live console line
        mark = "PASS" if row.get("pass") else "FAIL"
        rl = "R+" if retr_hit else ("R0" if row["no_retrieval"] else "R-")
        extra = ""
        if do_gen:
            if row["no_retrieval"] and not row.get("generated"):
                extra = " |  --- REFUSED (no LLM call) ---"
            elif row["no_retrieval"]:
                extra = " |  --- EMPTY CTX -> LLM CALLED ---"
            else:
                ov = row["overflow"]
                extra = f" | {row['prompt_tokens']:>4}tok" + (f" OVER+{ov}" if ov else "        ")
                if do_judge:
                    extra += f" | {row.get('judge','')[:9]:<9}"
        print(f"  [{i:>2}/{len(questions)}] {mark:<4} {rl} {row['id']:<6} "
              f"ctx={ctx_recall:>4.0%}{extra}  {q['query'][:44]}")

    write_report(args, gold, rows, pipe, budget, do_gen, do_judge)
    summarize(rows, do_gen, do_judge, budget, pipe.llm.prompt_budget())


def agg(rows, key, pred=None):
    vals = [r for r in rows if pred is None or pred(r)]
    if not vals:
        return 0.0
    return sum(float(r.get(key) or 0) for r in vals) / len(vals)


def summarize(rows, do_gen, do_judge, budget, client_budget=None):
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unans = [r for r in rows if r["category"] == "unanswerable"]

    print(f"\n{'='*78}\n  AGGREGATE\n{'='*78}")
    print(f"  RETRIEVAL")
    print(f"    retrieval hit rate (all facts in context) : "
          f"{sum(r['retrieval_hit'] for r in answerable)}/{len(answerable)}"
          f"  ({agg(answerable,'retrieval_hit')*100:.0f}%)")
    print(f"    mean context recall                       : {agg(answerable,'context_recall')*100:.0f}%")
    print(f"    document routing accuracy                 : "
          f"{sum(r['doc_hit'] for r in answerable)}/{len(answerable)}")

    no_retr = [r for r in rows if r.get("no_retrieval")]
    print(f"\n  GROUNDING")
    print(f"    queries with ZERO chunks retrieved        : {len(no_retr)}/{len(rows)}"
          f"   {sorted(r['id'] for r in no_retr)}")
    ungrounded = [r for r in rows if r.get("generated") and r["n_retrieved"] == 0]
    flag = "  <-- ungrounded: answer NOT from documents" if ungrounded else "  <-- none. good."
    print(f"    LLM invoked with EMPTY context            : {len(ungrounded)}/{len(rows)}{flag}")
    # Of those, how many did the model actually answer vs. abstain on its own?
    # Abstention here is emergent (system-prompt goodwill), NOT guaranteed.
    if ungrounded:
        spoke = [r for r in ungrounded if not r.get("abstained")]
        print(f"      of which model ANSWERED anyway          : {len(spoke)}"
              f"   {sorted(r['id'] for r in spoke)}")
        print(f"      of which model abstained on its own    : {len(ungrounded)-len(spoke)}"
              f"   (emergent, not guaranteed)")
    nr_unans = [r for r in no_retr if r["category"] == "unanswerable"]
    nr_ans = [r for r in no_retr if r["category"] != "unanswerable"]
    print(f"      of which correct (unanswerable)         : {len(nr_unans)}")
    print(f"      of which retrieval failures (answerable): {len(nr_ans)}"
          f"   {sorted(r['id'] for r in nr_ans)}")

    if do_gen:
        graded = [r for r in answerable if r.get("generated")]
        print(f"\n  ANSWER  (grounded answers only — {len(graded)}/{len(answerable)} answerable queries)")
        print(f"    mean fact recall in answer                : {agg(graded,'fact_recall')*100:.0f}%")
        if do_judge:
            for v in ("CORRECT", "PARTIAL", "INCORRECT"):
                n = sum(1 for r in graded if r.get("judge") == v)
                print(f"    judge {v:<9}                           : {n}/{len(graded)}")
        print(f"    correct abstentions (unanswerable)        : "
              f"{sum(r['abstained'] for r in unans)}/{len(unans)}")
        halluc = [r for r in unans if not r["abstained"]]
        if halluc:
            print(f"    !! HALLUCINATED on unanswerable           : {[r['id'] for r in halluc]}")

        print(f"\n  CONTEXT WINDOW (usable budget: {budget} tokens)")
        over = [r for r in rows if r.get("overflow", 0) > 0]
        pts = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
        if pts:
            print(f"    prompt tokens  min/mean/max               : "
                  f"{min(pts)} / {sum(pts)//len(pts)} / {max(pts)}")
        print(f"    queries OVERFLOWING the window            : {len(over)}/{len(rows)}")
        if over:
            print(f"    worst overflow                            : "
                  f"+{max(r['overflow'] for r in over)} tokens (silently truncated)")

        print(f"\n  TOKEN BUDGETER  (client budget: {client_budget} tokens)")
        dropped = [r for r in rows if r.get("chunks_dropped")]
        print(f"    queries where chunks were dropped         : {len(dropped)}/{len(rows)}")
        if dropped:
            print(f"    chunks dropped (lowest-ranked first)      : "
                  f"{sum(r['chunks_dropped'] for r in dropped)} total, "
                  f"worst {max(r['chunks_dropped'] for r in dropped)} on one query")

        # Safety property. An under-count means the estimator lied and the prompt
        # can silently overflow again — the whole approach rests on this.
        graded_gen = [r for r in rows if r.get("generated") and r.get("prompt_tokens")]
        under = [r for r in graded_gen if r.get("est_undercount")]
        if graded_gen:
            ratios = [r["est_tokens"] / r["prompt_tokens"] for r in graded_gen]
            print(f"    estimate / actual  min/mean/max           : "
                  f"{min(ratios):.2f} / {sum(ratios)/len(ratios):.2f} / {max(ratios):.2f}")
        if under:
            print(f"    !! ESTIMATOR UNDER-COUNTED               : {len(under)}/{len(graded_gen)}"
                  f"  {[r['id'] for r in under]}  <-- UNSAFE, budget can overflow")
        else:
            print(f"    estimator never under-counted             : "
                  f"{len(graded_gen)}/{len(graded_gen)}  <-- safe")

        print(f"\n  LATENCY")
        print(f"    mean retrieval                            : {agg(rows,'retrieval_latency')*1000:.0f} ms")
        print(f"    mean generation                           : {agg(rows,'gen_latency'):.1f} s")

    n_pass = sum(1 for r in rows if r.get("pass"))
    print(f"\n{'-'*78}")
    print(f"  OVERALL: {n_pass}/{len(rows)} passed  ({n_pass/len(rows)*100:.0f}%)")
    print(f"{'-'*78}")

    print("\n  BY CATEGORY")
    for cat in ["easy_factual", "precise_lookup", "multi_hop", "follow_up", "distractor", "unanswerable"]:
        c = [r for r in rows if r["category"] == cat]
        if not c:
            continue
        p = sum(1 for r in c if r.get("pass"))
        rh = f"  retr {agg(c,'retrieval_hit')*100:>3.0f}%" if cat != "unanswerable" else ""
        print(f"    {cat:<16} {p}/{len(c)} pass ({p/len(c)*100:>3.0f}%){rh}")
    print()


def write_report(args, gold, rows, pipe, budget, do_gen, do_judge):
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unans = [r for r in rows if r["category"] == "unanswerable"]
    n_pass = sum(1 for r in rows if r.get("pass"))

    L = []
    L.append(f"# RAG Eval Report — `{args.label}`\n")
    L.append("Generated by `eval/run_eval.py`. No pipeline code was modified; this measures the system as it ships.\n")
    L.append("## Configuration\n")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append(f"| embedding model | `{pipe.vector_store.model_name}` |")
    L.append(f"| llm | `{pipe.llm.model}` |")
    L.append(f"| chunk size / overlap | {pipe.processor.chunk_tokens} tokens / "
             f"{pipe.processor.chunk_overlap_tokens} tokens |")
    L.append(f"| encoder token limit | {pipe.vector_store.get_max_seq_length()} |")
    L.append(f"| top_k | {pipe.top_k} |")
    L.append(f"| score_threshold | {pipe.score_threshold} |")
    L.append(f"| num_ctx / num_predict | {pipe.llm.num_ctx} / {pipe.llm.num_predict} |")
    L.append(f"| usable prompt budget | **{budget} tokens** |")
    L.append(f"| indexed chunks | {pipe.chunk_count()} |")
    L.append("")

    L.append("## Headline\n")
    L.append(f"- **Overall: {n_pass}/{len(rows)} passed ({n_pass/len(rows)*100:.0f}%)**")
    L.append(f"- Retrieval hit rate: **{agg(answerable,'retrieval_hit')*100:.0f}%** "
             f"({sum(r['retrieval_hit'] for r in answerable)}/{len(answerable)})")
    L.append(f"- Mean context recall: **{agg(answerable,'context_recall')*100:.0f}%**")
    no_retr = [r for r in rows if r.get("no_retrieval")]
    ungrounded = [r for r in rows if r.get("generated") and r["n_retrieved"] == 0]
    if do_gen:
        over = [r for r in rows if r.get("overflow", 0) > 0]
        graded = [r for r in answerable if r.get("generated")]
        L.append(f"- Mean fact recall (grounded answers only): **{agg(graded,'fact_recall')*100:.0f}%**")
        L.append(f"- Queries silently overflowing the context window: **{len(over)}/{len(rows)}**")
        L.append(f"- Correct abstentions: **{sum(r['abstained'] for r in unans)}/{len(unans)}**")
    L.append(f"- No-retrieval refusals (LLM never called): **{len(no_retr)}/{len(rows)}** "
             f"— `{'`, `'.join(sorted(r['id'] for r in no_retr))}`")
    L.append(f"- **LLM invoked with an EMPTY context: {len(ungrounded)}/{len(rows)}** "
             f"{'⚠️ silent hallucination risk' if ungrounded else '— none'}")
    L.append("")

    L.append("## Per-category\n")
    L.append("| category | pass | retrieval hit | mean ctx recall |")
    L.append("|---|---|---|---|")
    for cat in ["easy_factual", "precise_lookup", "multi_hop", "follow_up", "distractor", "unanswerable"]:
        c = [r for r in rows if r["category"] == cat]
        if not c:
            continue
        p = sum(1 for r in c if r.get("pass"))
        if cat == "unanswerable":
            L.append(f"| {cat} | {p}/{len(c)} | — | — |")
        else:
            L.append(f"| {cat} | {p}/{len(c)} | {agg(c,'retrieval_hit')*100:.0f}% | {agg(c,'context_recall')*100:.0f}% |")
    L.append("")

    L.append("## Per-query\n")
    hdr = "| id | cat | pass | retr | ctx recall | docs |"
    sep = "|---|---|---|---|---|---|"
    if do_gen:
        hdr = "| id | cat | pass | retr | ctx recall | fact recall | prompt tok | judge | docs |"
        sep = "|---|---|---|---|---|---|---|---|---|"
    L.append(hdr)
    L.append(sep)
    for r in rows:
        p = "PASS" if r.get("pass") else "**FAIL**"
        rh = "yes" if r["retrieval_hit"] else "**no**"
        docs = ", ".join(r["docs"]) or "—"
        if do_gen:
            ov = f" **(+{r['overflow']} OVER)**" if r.get("overflow") else ""
            L.append(f"| {r['id']} | {r['category']} | {p} | {rh} | {r['context_recall']*100:.0f}% | "
                     f"{r.get('fact_recall',0)*100:.0f}% | {r.get('prompt_tokens','—')}{ov} | "
                     f"{r.get('judge','—')} | {docs} |")
        else:
            L.append(f"| {r['id']} | {r['category']} | {p} | {rh} | {r['context_recall']*100:.0f}% | {docs} |")
    L.append("")

    fails = [r for r in rows if not r.get("pass")]
    if fails:
        L.append("## Failures in detail\n")
        for r in fails:
            L.append(f"### {r['id']} — {r['category']}\n")
            L.append(f"**Q:** {r['query']}\n")
            if r["missing_from_context"]:
                L.append(f"**Facts never retrieved into context:** `{'`, `'.join(r['missing_from_context'])}`\n")
            L.append(f"**Retrieved from:** {', '.join(r['docs']) or '(nothing)'} "
                     f"— top score {r['top_score']}\n")
            if do_gen:
                if r.get("overflow"):
                    L.append(f"**Context overflow:** prompt was {r['prompt_tokens']} tokens vs "
                             f"{budget} usable — **{r['overflow']} tokens silently dropped by Ollama**\n")
                if r.get("judge"):
                    L.append(f"**Judge:** {r['judge']} — {r.get('judge_reason','')}\n")
                L.append(f"**Answer:**\n\n> {(r.get('answer') or '').strip()[:600]}\n")
            L.append("")

    open(args.out, "w").write("\n".join(L))
    print(f"\n  Report written to {args.out}")


if __name__ == "__main__":
    main()
