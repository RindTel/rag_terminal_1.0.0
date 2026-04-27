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
        store_dir: str = "vectorstore",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            store_dir:   Directory where the FAISS index and metadata are stored.
            model_name:  sentence-transformers model to use for embeddings.
        """
        self.store_dir  = store_dir
        self.model_name = model_name
        self._model     = None     
        self._index     = None     
        self._chunks: List[dict] = []  

        os.makedirs(self.store_dir, exist_ok=True)

    # Embedding 

    def _get_model(self):
        """Load (or reuse) the embedding model."""
        if self._model is None:
            SentenceTransformer = _load_sentence_transformers()
            logger.info(f"Loading embedding model '{self.model_name}' ...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model ready.")
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of strings into embedding vectors.
        Returns shape (N, embedding_dim) float32 numpy array.
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True, 
        )
        return embeddings.astype("float32")

    # Index management 

    def _get_or_create_index(self, dim: int):
        """Create a FAISS IndexFlatIP (inner product on normalised vecs = cosine)."""
        faiss = _load_faiss()
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
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
                "doc_id":      chunk.doc_id,
            })

        logger.info(f"Index now contains {index.ntotal} vectors.")
        return len(chunks)

    def remove_file(self, filename: str):
        """
        Remove all chunks belonging to a specific source file.
        FAISS does not support in-place deletion, so we rebuild the index.
        """
        kept = [c for c in self._chunks if c["source_file"] != filename]
        if len(kept) == len(self._chunks):
            logger.warning(f"No chunks found for '{filename}' — nothing removed.")
            return

        removed = len(self._chunks) - len(kept)
        logger.info(f"Removing {removed} chunks for '{filename}', rebuilding index ...")

        if not kept:
            self._index  = None
            self._chunks = []
            return

        texts = [c["text"] for c in kept]
        embeddings = self.embed_texts(texts)
        faiss = _load_faiss()
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._chunks = kept
        logger.info(f"Index rebuilt with {self._index.ntotal} vectors.")

    def clear(self):
        """Wipe the entire index."""
        self._index  = None
        self._chunks = []
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

    def save(self):
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            logger.info("Nothing to save — index is empty.")
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

            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    self.model_name = cfg.get("model_name", self.model_name)

            logger.info(
                f"Vector store loaded: {self._index.ntotal} vectors, "
                f"{len(self.get_indexed_files())} file(s)."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self._index  = None
            self._chunks = []
            return False
