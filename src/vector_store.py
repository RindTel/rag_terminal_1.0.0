"""
src/vector_store.py
───────────────────
Manages embedding generation and FAISS vector storage.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  - Only 22M parameters — fast and memory-efficient
  - 384-dimensional embeddings
  - Works great for semantic similarity / Q&A retrieval
  - No GPU required; runs comfortably on CPU

FAISS index type: IndexFlatL2 (exact nearest-neighbor, L2 distance)
  - Simple and reliable for up to ~50k chunks
  - No approximate index needed at this scale
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional

import config

logger = logging.getLogger(__name__)

_faiss = None
_SentenceTransformer = None

def _load_faiss():
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS is required.\n"
                "Install with:  pip install faiss-cpu"
            )
    return _faiss

def _load_sentence_transformers():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install with:  pip install sentence-transformers"
            )
    return _SentenceTransformer

# VectorStore class

class VectorStore:
    """
    Wraps a FAISS index and a metadata store.
    Provides embed, add, search, save, and load operations.
    """

    INDEX_FILE    = "faiss.index"
    METADATA_FILE = "metadata.pkl"
    CONFIG_FILE   = "config.json"

    def __init__(
        self,
        store_dir: str = None,
        model_name: str = None,
    ):
        """
        Args:
            store_dir:   Directory where the FAISS index and metadata are stored.
            model_name:  sentence-transformers model to use for embeddings.
                         Defaults to config.EMBEDDING_MODEL.
        """
        self.store_dir  = store_dir if store_dir is not None else config.VECTORSTORE_DIR

        # What the user ASKED for, vs what the loaded index was actually built
        # with. These diverge when EMBEDDING_MODEL changes but the old index is
        # still on disk — see load().
        self._configured_model = (
            model_name if model_name is not None else config.EMBEDDING_MODEL
        )
        self.model_name    = self._configured_model
        self.model_mismatch = False

        self._model     = None
        self._index     = None
        self._chunks: List[dict] = []

        # Bumped on every mutation of _chunks. A derived index (e.g. the BM25
        # keyword index in HybridRetriever) reads this to know when to rebuild,
        # so it can never silently drift out of sync with the chunk list.
        self._version   = 0

        os.makedirs(self.store_dir, exist_ok=True)

    # Embedding 

    def _get_model(self):
        """Load (or reuse) the embedding model."""
        if self._model is None:
            SentenceTransformer = _load_sentence_transformers()
            logger.info(f"Loading embedding model '{self.model_name}' (CPU) ...")
            # CPU on purpose: the GPU is occupied by the local Ollama model (a
            # 4 GB card is nearly full with Qwen), and putting the encoder there
            # too causes CUDA OOM. all-MiniLM is tiny and fast on CPU; keeping it
            # off the GPU leaves that memory for generation, the real bottleneck.
            self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Embedding model ready.")
        return self._model

    def get_tokenizer(self):
        """
        The embedding model's own tokenizer. The chunker must size chunks with
        THIS tokenizer — anything else is a guess, and a wrong guess is silent
        truncation at encode time.
        """
        return self._get_model().tokenizer

    def get_max_seq_length(self) -> int:
        """
        Hard token ceiling of the encoder. Text beyond this is silently dropped
        by model.encode() — it does not raise, it just stops looking.
        """
        return self._get_model().max_seq_length

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of strings into embedding vectors.
        Returns shape (N, embedding_dim) float32 numpy array.
        """
        model = self._get_model()

        # Defence in depth. encode() truncates past max_seq_length WITHOUT
        # warning, which is how 66% of the corpus went unembedded. If oversized
        # text ever reaches this point again, it says so out loud.
        self._warn_if_truncating(texts)

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def _warn_if_truncating(self, texts: List[str]):
        """Log a WARNING for any text the encoder will silently cut short."""
        try:
            tokenizer = self.get_tokenizer()
            limit = self.get_max_seq_length()
        except Exception:
            return  # never let a diagnostic break embedding

        oversized = []
        for t in texts:
            n = len(tokenizer(t, add_special_tokens=True, verbose=False)["input_ids"])
            if n > limit:
                oversized.append(n)

        if oversized:
            logger.warning(
                "EMBEDDING TRUNCATION: %d/%d text(s) exceed the encoder limit of %d "
                "tokens (largest: %d). Everything past %d tokens will NOT be embedded "
                "and cannot be retrieved.",
                len(oversized), len(texts), limit, max(oversized), limit,
            )

    # Index management 

    def _get_or_create_index(self, dim: int):
        """Create a FAISS IndexFlatIP (inner product on normalised vecs = cosine)."""
        faiss = _load_faiss()
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        elif self._index.d != dim:
            # Swapping EMBEDDING_MODEL to one with a different vector size. FAISS
            # would raise something opaque here; say what actually went wrong.
            raise ValueError(
                f"Embedding dimension mismatch: the index in '{self.store_dir}' holds "
                f"{self._index.d}-dim vectors, but '{self.model_name}' produces {dim}-dim "
                f"ones. An index cannot mix dimensions — clear it and re-index "
                f"(>> REINDEX) to rebuild with this model."
            )
        return self._index

    def add_chunks(self, chunks) -> int:
        """
        Embed and add a list of DocumentChunk objects to the index.
        Returns the number of chunks added.

        Each chunk is stored as a metadata dict alongside its vector.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do.")
            return 0

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks ...")
        embeddings = self.embed_texts(texts)

        index = self._get_or_create_index(embeddings.shape[1])
        index.add(embeddings)

        for chunk in chunks:
            self._chunks.append({
                "text":        chunk.text,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "total_chunks":chunk.total_chunks,
                "page_number": chunk.page_number,
                "chunk_id":    chunk.chunk_id,
            })

        self._version += 1
        logger.info(f"Index now contains {index.ntotal} vectors.")
        return len(chunks)

    def upsert_chunks(self, chunks) -> int:
        """
        Add chunks, REPLACING any existing chunks from the same source file.

        This is what ingestion should call. Plain add_chunks() appends
        unconditionally, so re-indexing a file used to stack a second copy of its
        chunks on top of the first.

        Keyed on source_file — the identity of "the document being replaced".
        NOT on chunk_id: that is a hash of the chunk's own text, so the moment a
        document is edited (the whole reason to re-index) every chunk_id changes
        and nothing would match. The old chunks would survive as orphans and the
        index would serve stale and fresh content side by side.
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty list — nothing to do.")
            return 0

        incoming = {c.source_file for c in chunks}
        replaced = sum(1 for c in self._chunks if c["source_file"] in incoming)
        if replaced:
            logger.info(
                "Upsert: replacing %d existing chunk(s) for %s",
                replaced, ", ".join(sorted(incoming)),
            )
            self.remove_files(incoming)

        return self.add_chunks(chunks)

    def remove_files(self, filenames) -> int:
        """
        Remove every chunk belonging to any of `filenames`. Returns the count removed.
        FAISS has no in-place delete, so the index is rebuilt around the survivors.
        """
        filenames = set(filenames)
        keep_idx = [
            i for i, c in enumerate(self._chunks) if c["source_file"] not in filenames
        ]
        removed = len(self._chunks) - len(keep_idx)

        if not removed:
            logger.warning(
                "No chunks found for %s — nothing removed.", ", ".join(sorted(filenames))
            )
            return 0

        logger.info("Removing %d chunk(s), rebuilding index ...", removed)
        self._rebuild_from_kept(keep_idx)
        return removed

    def remove_file(self, filename: str):
        """Remove all chunks belonging to one source file."""
        self.remove_files({filename})

    def _rebuild_from_kept(self, keep_idx: List[int]):
        """
        Rebuild the index around the chunks at `keep_idx`.

        Reuses the vectors already stored in FAISS via reconstruct() rather than
        re-running the encoder. The previous implementation re-embedded every
        surviving chunk, which made removing one file an O(entire corpus) job —
        deleting from a 10k-chunk index meant 10k forward passes. The vectors are
        right there; recomputing them was pure waste.
        """
        faiss = _load_faiss()
        self._version += 1

        if not keep_idx:
            # Nothing survives. save() persists this as a purge of the on-disk store.
            self._index = None
            self._chunks = []
            logger.info("Index is now empty.")
            return

        kept_vectors = np.vstack(
            [self._index.reconstruct(int(i)) for i in keep_idx]
        ).astype("float32")

        self._index = faiss.IndexFlatIP(kept_vectors.shape[1])
        self._index.add(kept_vectors)
        self._chunks = [self._chunks[i] for i in keep_idx]

        logger.info(f"Index rebuilt with {self._index.ntotal} vectors (no re-embedding).")

    def clear(self):
        """
        Wipe the entire index.

        Also snaps back to the CONFIGURED embedding model and drops the cached
        encoder. This is what makes >> REINDEX actually adopt a changed
        EMBEDDING_MODEL — without it, a rebuild would silently re-embed with the
        old model the stale index happened to be using.
        """
        self._index  = None
        self._chunks = []
        self._version += 1

        if self.model_name != self._configured_model:
            logger.info(
                "Rebuilding with the configured embedding model '%s' (was '%s').",
                self._configured_model, self.model_name,
            )
            self.model_name = self._configured_model
            self._model = None          # force a reload of the new encoder

        self.model_mismatch = False
        logger.info("Vector store cleared.")

    # Search

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[dict, float]]:
        """
        Semantic search: return the top_k most relevant chunks.

        Args:
            query:           The user's question.
            top_k:           How many chunks to return.
            score_threshold: Minimum cosine similarity (0–1). Chunks below
                             this score are filtered out.

        Returns:
            List of (chunk_metadata_dict, score) tuples, sorted by score desc.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search called on empty index.")
            return []

        query_vec = self.embed_texts([query])  # shape (1, dim)
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue 
            if float(score) < score_threshold:
                continue
            results.append((self._chunks[idx], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_indexed_files(self) -> List[str]:
        """Return a deduplicated list of source filenames currently in the index."""
        return list({c["source_file"] for c in self._chunks})

    def chunk_count(self) -> int:
        """Total number of chunks in the index."""
        return len(self._chunks)

    # Persistence 

    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def _purge_disk(self):
        """
        Remove the persisted store.

        An empty store means NO FILES ON DISK. This is what makes deletion
        durable: load() already treats missing files as "start fresh", which is
        exactly the right meaning for an empty index.
        """
        removed = []
        for name in (self.INDEX_FILE, self.METADATA_FILE, self.CONFIG_FILE):
            path = os.path.join(self.store_dir, name)
            if os.path.exists(path):
                os.remove(path)
                removed.append(name)

        if removed:
            logger.info(
                "Vector store is empty — removed persisted files from '%s': %s",
                self.store_dir, ", ".join(removed),
            )

    def save(self):
        """Persist the FAISS index and metadata to disk."""
        # An empty index is not "nothing to save" — it is a DELETION to persist.
        # Returning early here is what let a deleted document survive on disk and
        # come back on the next launch.
        if self.is_empty():
            self._purge_disk()
            return

        faiss = _load_faiss()
        index_path    = os.path.join(self.store_dir, self.INDEX_FILE)
        metadata_path = os.path.join(self.store_dir, self.METADATA_FILE)
        config_path   = os.path.join(self.store_dir, self.CONFIG_FILE)

        faiss.write_index(self._index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self._chunks, f)

        with open(config_path, "w") as f:
            json.dump({"model_name": self.model_name}, f)

        logger.info(f"Vector store saved to '{self.store_dir}' ({self._index.ntotal} vectors).")

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.
        Returns True if successful, False if no saved state exists.
        """
        index_path    = os.path.join(self.store_dir, self.INDEX_FILE)
        metadata_path = os.path.join(self.store_dir, self.METADATA_FILE)
        config_path   = os.path.join(self.store_dir, self.CONFIG_FILE)

        if not all(os.path.exists(p) for p in [index_path, metadata_path]):
            logger.info("No saved vector store found — starting fresh.")
            return False

        try:
            faiss = _load_faiss()
            self._index = faiss.read_index(index_path)

            with open(metadata_path, "rb") as f:
                self._chunks = pickle.load(f)

            # The index and the metadata are two separate files written in
            # sequence. If they ever desync (a crash between the two writes),
            # search() would index self._chunks[idx] out of a mismatched list and
            # return THE WRONG CHUNK with a confident score — citing a document
            # that never contained the answer. Refuse to serve that.
            if self._index.ntotal != len(self._chunks):
                logger.error(
                    "CORRUPT VECTOR STORE in '%s': %d vectors but %d metadata chunks. "
                    "Refusing to load — retrieval would cite the wrong documents. "
                    "Re-index to rebuild.",
                    self.store_dir, self._index.ntotal, len(self._chunks),
                )
                self._index = None
                self._chunks = []
                return False

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                persisted = cfg.get("model_name", self._configured_model)

                # The index's vectors were produced by `persisted`. Querying them
                # with a different encoder is meaningless (and a dimension change
                # would crash FAISS outright), so keep using `persisted` and say
                # so LOUDLY — a silent revert here is what made EMBEDDING_MODEL
                # look configurable while being ignored.
                if persisted != self._configured_model:
                    self.model_mismatch = True
                    logger.warning(
                        "EMBEDDING MODEL MISMATCH: config asks for '%s', but the index in "
                        "'%s' was built with '%s'. Using '%s' so the existing index stays "
                        "queryable. Your configured model is NOT in effect — click "
                        ">> REINDEX to rebuild the index with '%s'.",
                        self._configured_model, self.store_dir, persisted,
                        persisted, self._configured_model,
                    )
                self.model_name = persisted

            self._version += 1
            logger.info(
                f"Vector store loaded: {self._index.ntotal} vectors, "
                f"{len(self.get_indexed_files())} file(s)."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self._index  = None
            self._chunks = []
            self._version += 1
            return False
