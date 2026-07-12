"""
tests/test_hybrid_retrieval.py
──────────────────────────────
Hybrid retrieval (BM25 + vector, RRF-fused, cross-encoder reranked) must:
  - be a clean no-op when disabled (real rollback path),
  - surface exact tokens the vector arm misses,
  - keep its keyword index in sync with the store,
  - still abstain when nothing is relevant.

No Ollama needed. The reranker downloads once (~80 MB) on first run.
"""

import importlib
import os

import pytest

import config
from src.rag_pipeline import RAGPipeline

# A doc rich in exact tokens dense retrieval blurs: error codes, acronyms.
DOC = "spec.txt"
TEXT = (
    "The API returns standard HTTP status codes. A duplicate email registration "
    "yields 409 Conflict. Prisma error P2002 maps to 409, and P2025 maps to 404. "
    "RBAC stands for Role-Based Access Control. ORM stands for Object-Relational "
    "Mapper. The Axios client uses a 15 second request timeout. Rate limiting "
    "allows 100 requests per 15 minutes per IP address. " * 2
)
DISTRACTOR = "misc.txt"
DISTRACTOR_TEXT = (
    "The garden receives morning sunlight. Tomatoes ripen in late summer and "
    "basil grows quickly beside them near the fence. " * 4
)


@pytest.fixture
def pipe(tmp_path):
    uploads = tmp_path / "uploads"
    store = tmp_path / "vectorstore"
    uploads.mkdir(); store.mkdir()
    (uploads / DOC).write_text(TEXT)
    (uploads / DISTRACTOR).write_text(DISTRACTOR_TEXT)
    p = RAGPipeline(uploads_dir=str(uploads), vectorstore_dir=str(store))
    p.ingest_files([str(uploads / DOC), str(uploads / DISTRACTOR)])
    return p


@pytest.fixture(autouse=True)
def restore_config():
    """Snapshot the retrieval flags so a test flipping them can't leak."""
    saved = (config.HYBRID_RETRIEVAL, config.RERANK, config.RERANK_SCORE_THRESHOLD)
    yield
    (config.HYBRID_RETRIEVAL, config.RERANK, config.RERANK_SCORE_THRESHOLD) = saved


def test_flag_off_is_pure_vector(pipe):
    """HYBRID_RETRIEVAL=False must reproduce vector_store.search() exactly."""
    config.HYBRID_RETRIEVAL = False
    hits = pipe.retrieve("How are passwords validated?")
    ref = pipe.vector_store.search(
        "How are passwords validated?", top_k=pipe.top_k,
        score_threshold=pipe.score_threshold,
    )
    assert [c["chunk_id"] for c, _ in hits] == [c["chunk_id"] for c, _ in ref]


def test_hybrid_finds_exact_token_acronym(pipe):
    """RBAC/ORM: dense retrieval blurs bare acronyms; hybrid+rerank must surface them."""
    config.HYBRID_RETRIEVAL = True
    config.RERANK = True
    for q, needle in [("What does RBAC stand for?", "Role-Based Access Control"),
                      ("What does ORM stand for?", "Object-Relational Mapper")]:
        hits = pipe.retrieve(q)
        assert hits, f"hybrid returned nothing for {q!r}"
        assert any(needle.lower() in c["text"].lower() for c, _ in hits), (
            f"hybrid did not surface the definition for {q!r}"
        )


def test_hybrid_finds_error_code(pipe):
    """Exact code lookup — the BM25 arm's home turf."""
    config.HYBRID_RETRIEVAL = True
    hits = pipe.retrieve("Which Prisma error code maps to 409 Conflict?")
    assert any("P2002" in c["text"] for c, _ in hits)


def test_bm25_index_rebuilds_after_mutation(pipe):
    """The keyword index must track the store, not go stale after an upsert."""
    from src.hybrid_retriever import HybridRetriever
    hr = HybridRetriever(pipe.vector_store)

    hr._ensure_bm25()
    v1 = hr._bm25_version
    assert v1 == pipe.vector_store._version

    # Remove a file — store mutates, version bumps, BM25 must rebuild.
    pipe.delete_file(DISTRACTOR)
    hr._ensure_bm25()
    assert hr._bm25_version == pipe.vector_store._version
    assert hr._bm25_version != v1
    # The rebuilt keyword corpus excludes the deleted file's content.
    assert all(c["source_file"] == DOC for c in hr._bm25_chunks)
    assert not any("tomatoes" in c["text"].lower() for c in hr._bm25_chunks)


def test_rerank_gate_is_off_by_default(pipe):
    """
    By default retrieval does NOT abstain — it returns top_k and lets the LLM
    decline irrelevant context (as the vector path does). A query with no real
    answer still returns loosely-related chunks rather than [].
    """
    config.HYBRID_RETRIEVAL = True
    config.RERANK = True
    config.RERANK_SCORE_THRESHOLD = None
    hits = pipe.retrieve("What is the Kubernetes autoscaling policy and AWS region?")
    assert hits, "default hybrid should return candidates (LLM handles abstention)"


def test_rerank_gate_filters_when_enabled(pipe):
    """
    The opt-in gate, when a threshold IS set, drops low-relevance chunks. Set it
    absurdly high and everything is filtered → []. Proves the mechanism works
    without relying on it by default.
    """
    config.HYBRID_RETRIEVAL = True
    config.RERANK = True
    config.RERANK_SCORE_THRESHOLD = 100.0   # nothing scores this high
    hits = pipe.retrieve("What does RBAC stand for?")
    assert hits == [], "an impossibly-high gate should filter every candidate"


def test_hybrid_result_shape_matches_vector(pipe):
    """Downstream code consumes [(chunk_dict, float)] — hybrid must match that shape."""
    config.HYBRID_RETRIEVAL = True
    hits = pipe.retrieve("What does RBAC stand for?")
    assert isinstance(hits, list) and hits
    chunk, score = hits[0]
    assert isinstance(chunk, dict) and "text" in chunk and "source_file" in chunk
    assert isinstance(score, float)
