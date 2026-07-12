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
import logging

# Paths

UPLOADS_DIR     = os.getenv("QWENRAG_UPLOADS_DIR",     "uploads")
VECTORSTORE_DIR = os.getenv("QWENRAG_VECTORSTORE_DIR", "vectorstore")

# Ollama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")

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
RERANK           = _env_bool("RERANK", True)            # rerank fused candidates
RERANK_MODEL     = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
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
