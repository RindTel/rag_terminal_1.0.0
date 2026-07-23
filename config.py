"""
config.py
─────────
Central configuration for QwenRAG.

Every value here is a real default — the code reads them. Change a value (or set
the matching env var) and the app's behaviour changes. Nothing here is decorative.

Env vars override file defaults, so you can do:

    TOP_K=8 LOG_LEVEL=DEBUG streamlit run app.py
"""

import os
import sys
import logging

# Runtime mode
# ────────────
# "Frozen" == running inside the packaged desktop app (PyInstaller). In that mode
# the app is fully self-contained: an embedded llama.cpp engine instead of the
# Ollama daemon, ONNX fastembed instead of PyTorch, and every model bundled — no
# network. When NOT frozen (normal `streamlit run`, pytest, the eval harness)
# every default below is byte-identical to before, so the dev workflow and the
# committed eval baselines are untouched.

IS_FROZEN = bool(getattr(sys, "frozen", False))


def _resources_dir() -> str:
    """Directory holding bundled models + tokenizer (only meaningful when frozen)."""
    if IS_FROZEN:
        # PyInstaller onefile unpacks to _MEIPASS; onedir puts data beside the exe.
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "resources")


RESOURCES_DIR = os.getenv("QWENRAG_RESOURCES_DIR", _resources_dir())


def _default_data_dir() -> str:
    """
    Per-user writable dir for uploads + index.

    An installed app often lives in a read-only location (Program Files), so the
    bundle must NOT write beside its executable. In dev this returns "" and the
    paths below stay CWD-relative — exactly the original behaviour.
    """
    if IS_FROZEN:
        try:
            from platformdirs import user_data_dir
            return user_data_dir("QwenRAG", "QwenRAG")
        except Exception:
            # platformdirs missing is not fatal — fall back beside the executable.
            return os.path.join(os.path.dirname(sys.executable), "data")
    return ""

_DATA_DIR = _default_data_dir()

# Paths

UPLOADS_DIR     = os.getenv("QWENRAG_UPLOADS_DIR",     os.path.join(_DATA_DIR, "uploads")     if _DATA_DIR else "uploads")
VECTORSTORE_DIR = os.getenv("QWENRAG_VECTORSTORE_DIR", os.path.join(_DATA_DIR, "vectorstore") if _DATA_DIR else "vectorstore")

# When frozen, forbid accidental HuggingFace network access — everything is
# bundled. Not set in dev, so the normal HF cache/download path is unaffected.
if IS_FROZEN:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# LLM backend
#
# "ollama"   — talk to the Ollama daemon (dev default; what the eval + tests use).
# "llamacpp" — load a GGUF in-process via llama-cpp-python (the packaged app; no
#              external daemon). Auto-selected when frozen.

LLM_BACKEND = os.getenv("LLM_BACKEND", "llamacpp" if IS_FROZEN else "ollama")

# Ollama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")

# llama.cpp — path to the bundled GGUF used when LLM_BACKEND == "llamacpp".

LLAMACPP_MODEL_PATH = os.getenv(
    "LLAMACPP_MODEL_PATH",
    os.path.join(RESOURCES_DIR, "models", "qwen2.5-3b-instruct-q4_k_m.gguf"),
)

# LLM generation parameters

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))   # Low = factual
LLM_TOP_P       = float(os.getenv("LLM_TOP_P",       "0.9"))

# num_ctx is the WHOLE window — prompt AND generation share it. The prompt
# budgeter (OllamaClient.prompt_budget) derives its limit from these two, so
# raising num_ctx here is what lets more retrieved chunks reach the model.
LLM_NUM_CTX     = int(os.getenv("LLM_NUM_CTX",     "4096"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1024"))    # Max output tokens

# Embedding model
#
# The encoder's max_seq_length is a HARD ceiling — text beyond it is silently
# discarded at embed time, not truncated with a warning. CHUNK_TOKENS below must
# stay under it (DocumentProcessor raises if it doesn't).
#
#   all-MiniLM-L6-v2      384-dim, 256-token limit,  ~90 MB   (default)
#   BAAI/bge-small-en-v1.5  384-dim, 512-token limit          (drop-in, stronger)
#   all-mpnet-base-v2     768-dim, 384-token limit, ~420 MB
#
# Changing this INVALIDATES an existing index (different dimensions). The app
# warns and keeps using the persisted model until you hit >> REINDEX.

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Embedding / rerank backend.
#
# "sentence-transformers" — PyTorch (dev default; matches the committed eval).
# "fastembed"             — ONNX runtime, no PyTorch, so the packaged artifact is
#                           ~2 GB smaller. Auto-selected when frozen. Retrieval
#                           parity vs the PyTorch path is proven by re-running
#                           eval/run_eval.py against the golden set before ship.

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "fastembed" if IS_FROZEN else "sentence-transformers")

# fastembed can't hand over the model's HF fast tokenizer (which the chunker needs
# for offset mapping), so the fastembed path loads a standalone tokenizer.json.
# Prefers a bundled copy (offline); falls back to a one-time Hub download in dev.
EMBED_TOKENIZER_PATH = os.getenv(
    "QWENRAG_EMBED_TOKENIZER_PATH",
    os.path.join(RESOURCES_DIR, "tokenizer", "all-MiniLM-L6-v2", "tokenizer.json"),
)
# Hard token ceiling of the embedder (all-MiniLM-L6-v2 = 256). Only consulted by
# the fastembed path; the sentence-transformers path reads model.max_seq_length.
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "256"))

# Where the bundled fastembed ONNX models live. When this dir exists (the packaged
# app, populated by packaging/fetch_resources.py) fastembed loads from it offline;
# otherwise (dev) fastembed uses its own default cache and downloads on first use.
FASTEMBED_CACHE = os.getenv("QWENRAG_FASTEMBED_CACHE", os.path.join(RESOURCES_DIR, "fastembed"))

# Document chunking
#
# Sized in TOKENS (measured with the embedding model's own tokenizer), not words.
# Words were the original bug: 512-word chunks became ~875 tokens, and MiniLM
# embedded only the first 254 of them — so two thirds of every document sat in
# the index, unreachable by search.

CHUNK_TOKENS         = int(os.getenv("CHUNK_TOKENS",         "220"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "32"))

# Retrieval

TOP_K           = int(os.getenv("TOP_K",           "6"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))  # Min cosine similarity (vector-only path)

# Hybrid retrieval (BM25 + vector, fused with RRF, reranked by a cross-encoder)
#
# Vector-only retrieval is weak on exact tokens — error codes, acronyms, proper
# nouns — because dense embeddings blur them. The BM25 arm matches them
# literally; the reranker then promotes the truly relevant chunk. Set
# HYBRID_RETRIEVAL=false to fall back to the exact pure-vector path.

def _env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

HYBRID_RETRIEVAL = _env_bool("HYBRID_RETRIEVAL", True)   # master flag; False = vector-only

# Conversational query rewriting.
#
# A follow-up like "what about its limitations?" is embedded with "it" unresolved,
# so retrieval can't tell what it's about. When on, a referential query is
# rewritten into a standalone one (using recent turns) BEFORE retrieval. Gated on
# a follow-up signal — standalone questions are never rewritten, so they pay no
# latency. Set false to fall back to raw-query retrieval.
QUERY_REWRITE = _env_bool("QUERY_REWRITE", True)

# General-knowledge fallback.
#
# When the documents can't answer — nothing indexed, nothing retrieved, or the
# grounded answer abstains — fall back to the LLM's own general knowledge, clearly
# LABELLED as not-from-your-documents (see GENERAL_SYSTEM_PROMPT). Grounded answers
# with sufficient context are untouched. Set False for strict document-only mode.
GENERAL_FALLBACK = _env_bool("GENERAL_FALLBACK", True)
RERANK           = _env_bool("RERANK", True)            # rerank fused candidates
RERANK_MODEL     = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Same reranker, ONNX build — used by the fastembed backend (no PyTorch).
RERANK_MODEL_FASTEMBED = os.getenv("RERANK_MODEL_FASTEMBED", "Xenova/ms-marco-MiniLM-L-6-v2")
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", "20"))  # per-arm pool before fusion
RRF_K             = int(os.getenv("RRF_K", "60"))             # reciprocal-rank-fusion constant

# Optional abstention gate on the cross-encoder's relevance logit.
#
# OFF by default (None), and deliberately so: on real 220-token technical chunks
# the cross-encoder's absolute scores do NOT cleanly separate answerable from
# unanswerable — a correct answer can score -5 while an irrelevant chunk scores
# -2. Abstention is the LLM's job (system-prompt rule #2, proven 2/2 on the
# unanswerable eval), not retrieval's. Retrieval returns the top_k reranked
# chunks and lets the model decline if they don't actually answer.
#
# Set a float to re-enable the gate (chunks scoring below it are dropped) only
# if you have calibrated it against your own corpus.
_rst = os.getenv("RERANK_SCORE_THRESHOLD")
RERANK_SCORE_THRESHOLD = float(_rst) if _rst not in (None, "") else None

# UI

# Slider max. Safe at 10: the prompt budgeter drops the lowest-ranked chunks if
# they don't fit, rather than letting Ollama truncate the highest-ranked ones.
UI_MAX_TOP_K = int(os.getenv("UI_MAX_TOP_K", "10"))

# Logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def configure_logging():
    """
    Install a root log handler.

    Without this, every logger.warning() in the app goes nowhere — including the
    ones that matter most: NO_RETRIEVAL (the answer is not grounded in your
    documents), the prompt budgeter dropping chunks, and the embedding-truncation
    guard. Streamlit does not configure logging for you, so this must be called
    from each entry point.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    # These are chatty at INFO and drown out our own logs.
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
