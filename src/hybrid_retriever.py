"""
src/hybrid_retriever.py
───────────────────────
Hybrid retrieval: a BM25 keyword arm alongside the dense vector arm, fused with
Reciprocal Rank Fusion, then reranked by a cross-encoder.

Why: dense embeddings blur exact tokens — error codes (P2002), acronyms (RBAC,
ORM), proper nouns — so vector-only retrieval misses precise lookups even when
the answer is right there in the corpus. BM25 matches those literally. RRF
combines the two rankings without needing their (incomparable) score scales to
agree, and the cross-encoder then scores true query-passage relevance to both
order the finalists and decide when NOTHING is relevant (abstention).

This composes OVER the existing VectorStore — it does not replace it. Turn it off
(config.HYBRID_RETRIEVAL=False) and RAGPipeline uses the pure-vector path
unchanged.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional

import config

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


# Lazy singletons — loaded once, reused. The cross-encoder is ~80 MB; loading it
# per query would dominate latency.
_BM25 = None
_RERANKER = None


def _load_bm25_class():
    global _BM25
    if _BM25 is None:
        try:
            from rank_bm25 import BM25Okapi
            _BM25 = BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for hybrid retrieval.\n"
                "Install with:  pip install rank_bm25\n"
                "(or set HYBRID_RETRIEVAL=false to use the vector-only path)"
            )
    return _BM25


def _rerank_scores(query: str, texts: List[str]) -> List[float]:
    """
    Cross-encoder relevance scores for (query, text) pairs, higher = more relevant.

    Routes to the same MiniLM cross-encoder via whichever backend config selects:
    sentence-transformers (PyTorch, dev) or fastembed (ONNX, the bundle). Loaded
    once and cached — the CPU load dominates latency otherwise.
    """
    global _RERANKER
    if config.EMBED_BACKEND.lower() == "fastembed":
        if _RERANKER is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            from .embeddings import fastembed_kwargs
            logger.info("Loading reranker '%s' (ONNX, fastembed) ...", config.RERANK_MODEL_FASTEMBED)
            _RERANKER = TextCrossEncoder(model_name=config.RERANK_MODEL_FASTEMBED, **fastembed_kwargs())
            logger.info("Reranker ready.")
        return [float(s) for s in _RERANKER.rerank(query, texts)]

    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker '%s' (CPU, sentence-transformers) ...", config.RERANK_MODEL)
        # CPU on purpose: the local LLM occupies the GPU; the cross-encoder is tiny
        # and fast on CPU, and keeping it off-GPU avoids CUDA OOM.
        _RERANKER = CrossEncoder(config.RERANK_MODEL, device="cpu")
        logger.info("Reranker ready.")
    return [float(s) for s in _RERANKER.predict([(query, t) for t in texts])]


class HybridRetriever:
    """
    BM25 + vector, fused with RRF, reranked by a cross-encoder.

    Holds a reference to the live VectorStore. The BM25 index is derived from the
    store's chunks and rebuilt whenever the store's _version changes, so it can
    never drift out of sync with what FAISS holds.
    """

    def __init__(self, vector_store):
        self.vs = vector_store
        self._bm25 = None
        self._bm25_version = -1        # forces a build on first use
        self._bm25_chunks: List[dict] = []

    # BM25 arm

    def _ensure_bm25(self):
        """(Re)build the BM25 index iff the vector store has mutated since last build."""
        if self._bm25 is not None and self._bm25_version == self.vs._version:
            return
        chunks = self.vs._chunks
        if not chunks:
            self._bm25, self._bm25_chunks = None, []
            self._bm25_version = self.vs._version
            return
        BM25Okapi = _load_bm25_class()
        corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_chunks = list(chunks)          # snapshot aligned to this build
        self._bm25_version = self.vs._version
        logger.info("BM25 index built over %d chunks.", len(chunks))

    def _bm25_rank(self, query: str, k: int) -> List[int]:
        """Return indices (into _bm25_chunks) of the top-k BM25 matches, score desc."""
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Drop zero-score docs — a BM25 of 0 means no query term matched at all.
        return [i for i in ranked[:k] if scores[i] > 0.0]

    # Dense arm

    def _vector_rank(self, query: str, k: int) -> List[dict]:
        """Top-k dense candidates as chunk dicts (no cosine floor — fusion needs the pool)."""
        hits = self.vs.search(query, top_k=k, score_threshold=0.0)
        return [chunk for chunk, _ in hits]

    # Fusion + rerank

    @staticmethod
    def _chunk_key(c: dict) -> str:
        return c.get("chunk_id") or f'{c.get("source_file")}#{c.get("chunk_index")}'

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidates: Optional[int] = None,
    ) -> List[Tuple[dict, float]]:
        """
        Hybrid search. Returns [(chunk, score)] sorted best-first, at most top_k.

        score is the cross-encoder relevance logit when reranking is on, else the
        RRF score. Returns [] when nothing clears the relevance gate — which is
        how the pipeline's no-retrieval abstention is preserved.
        """
        n = candidates or config.HYBRID_CANDIDATES

        # Two rankings over the same corpus.
        bm25_idx = self._bm25_rank(query, n)
        bm25_chunks = [self._bm25_chunks[i] for i in bm25_idx]
        vec_chunks = self._vector_rank(query, n)

        if not bm25_chunks and not vec_chunks:
            return []

        # Reciprocal Rank Fusion. Each arm contributes 1/(RRF_K + rank). Score
        # scales (cosine vs BM25) never have to be reconciled — only ranks.
        rrf: Dict[str, float] = {}
        pool: Dict[str, dict] = {}
        for arm in (bm25_chunks, vec_chunks):
            for rank, chunk in enumerate(arm):
                key = self._chunk_key(chunk)
                rrf[key] = rrf.get(key, 0.0) + 1.0 / (config.RRF_K + rank)
                pool.setdefault(key, chunk)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
        fused_chunks = [pool[k] for k, _ in fused][:n]

        if not config.RERANK:
            return [(pool[k], rrf[k]) for k, _ in fused[:top_k]]

        return self._rerank(query, fused_chunks, top_k)

    def _rerank(self, query, chunks, top_k) -> List[Tuple[dict, float]]:
        """
        Cross-encoder rerank.

        Reorders the fused candidates by true (query, passage) relevance and
        returns the top_k. This is purely reordering — abstention is left to the
        LLM (see config.RERANK_SCORE_THRESHOLD for why the score is not a
        reliable retrieval-level gate on this corpus).
        """
        if not chunks:
            return []
        scores = _rerank_scores(query, [c["text"] for c in chunks])

        scored = sorted(
            zip(chunks, (float(s) for s in scores)),
            key=lambda cs: cs[1],
            reverse=True,
        )

        # Optional, opt-in relevance gate. Off by default (None).
        if config.RERANK_SCORE_THRESHOLD is not None:
            gated = [(c, s) for c, s in scored if s >= config.RERANK_SCORE_THRESHOLD]
            if len(gated) < len(scored):
                logger.info("Hybrid: %d/%d candidates cleared the rerank gate.",
                            len(gated), len(scored))
            scored = gated

        return scored[:top_k]
