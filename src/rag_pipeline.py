"""
src/rag_pipeline.py
───────────────────
The orchestrator that connects:
  DocumentProcessor  →  VectorStore  →  OllamaClient

This is the single class you call from the UI. It manages the full
RAG lifecycle: ingest documents, retrieve relevant chunks, generate answers.
"""

import os
import logging
import shutil
from typing import List, Dict, Optional, Tuple

from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import OllamaClient

logger = logging.getLogger(__name__)

# Result dataclass

class RAGResult:
    """Holds the answer and supporting evidence for a single query."""

    def __init__(
        self,
        answer: str,
        sources: List[Dict],
        query: str,
        retrieved_count: int,
    ):
        self.answer          = answer
        self.sources         = sources   # list of chunk metadata dicts
        self.query           = query
        self.retrieved_count = retrieved_count

    def __repr__(self):
        return f"<RAGResult query={self.query!r} sources={self.retrieved_count}>"

# Main pipeline

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Typical usage:
        pipeline = RAGPipeline()
        pipeline.load()                          # Load existing index from disk
        pipeline.ingest_files(["my_doc.pdf"])    # Add new documents
        result = pipeline.query("What is X?")    # Ask a question
    """

    def __init__(
        self,
        uploads_dir: str  = "uploads",
        vectorstore_dir: str = "vectorstore",
        llm_model: str    = "qwen2.5:7b",
        chunk_size: int   = 512,
        chunk_overlap: int = 64,
        top_k: int        = 5,
        score_threshold: float = 0.25,
    ):
        """
        Args:
            uploads_dir:      Where uploaded files are stored.
            vectorstore_dir:  Where the FAISS index is persisted.
            llm_model:        Ollama model to use for generation.
            chunk_size:       Words per chunk (512 ≈ 350–400 tokens for English).
            chunk_overlap:    Overlap between consecutive chunks.
            top_k:            Number of chunks to retrieve per query.
            score_threshold:  Minimum cosine similarity to include a chunk.
        """
        self.uploads_dir  = uploads_dir
        self.top_k        = top_k
        self.score_threshold = score_threshold

        os.makedirs(uploads_dir, exist_ok=True)

        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore(store_dir=vectorstore_dir)
        self.llm = OllamaClient(model=llm_model)

    # Ingestion 

    def ingest_files(
        self,
        file_paths: List[str],
        progress_callback=None,
    ) -> Tuple[int, List[str]]:
        """
        Parse, chunk, embed, and index the given files.

        Args:
            file_paths:        Absolute paths to PDF or TXT files.
            progress_callback: Optional callable(current, total, filename)
                               for reporting progress to the UI.

        Returns:
            (total_chunks_added, list_of_errors)
        """
        total_added = 0
        errors: List[str] = []

        for i, path in enumerate(file_paths):
            filename = os.path.basename(path)
            if progress_callback:
                progress_callback(i, len(file_paths), filename)

            try:
                chunks = self.processor.process_file(path)
                added  = self.vector_store.add_chunks(chunks)
                total_added += added
                logger.info(f"Ingested '{filename}': {added} chunks.")
            except Exception as e:
                msg = f"Error processing '{filename}': {e}"
                logger.error(msg)
                errors.append(msg)

      
        self.vector_store.save()

        if progress_callback:
            progress_callback(len(file_paths), len(file_paths), "done")

        return total_added, errors

    def reindex_all(self, progress_callback=None) -> Tuple[int, List[str]]:
        """
        Wipe the current index and re-process all files in uploads_dir.
        Useful if settings (chunk_size, model) have changed.
        """
        self.vector_store.clear()
        files = self._get_all_upload_paths()
        if not files:
            logger.warning("No files in uploads directory to reindex.")
            return 0, []
        return self.ingest_files(files, progress_callback)

    # Retrieval 

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Return the most relevant chunks for a query.

        Returns:
            List of (chunk_dict, score) sorted by relevance.
        """
        k = top_k or self.top_k
        return self.vector_store.search(
            query,
            top_k=k,
            score_threshold=self.score_threshold,
        )

    # Full RAG query

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
        stream: bool = False,
    ):
        """
        Full RAG query: retrieve → build prompt → generate answer.

        Args:
            question:     The user's question.
            chat_history: Previous chat turns for context.
            stream:       If True, returns a generator of text tokens.
                          If False, returns a RAGResult object.

        Returns:
            RAGResult (stream=False) or generator (stream=True).
            When streaming, call get_last_sources() afterwards.
        """
        if self.vector_store.chunk_count() == 0:
            if stream:
                def _no_docs():
                    yield "⚠️ No documents are indexed yet. Please upload and index some files first."
                return _no_docs()
            return RAGResult(
                answer="⚠️ No documents are indexed yet. Please upload and index some files first.",
                sources=[],
                query=question,
                retrieved_count=0,
            )

        hits = self.retrieve(question)
        chunks = [chunk for chunk, _ in hits]
        self._last_sources = chunks  # Cache for streaming mode

        prompt = self.llm.build_rag_prompt(
            question=question,
            retrieved_chunks=chunks,
            chat_history=chat_history,
        )

        if stream:
            return self.llm.generate_stream(prompt)

        answer = self.llm.generate(prompt)
        return RAGResult(
            answer=answer,
            sources=chunks,
            query=question,
            retrieved_count=len(chunks),
        )

    def get_last_sources(self) -> List[Dict]:
        """Return the chunks used in the most recent streaming query."""
        return getattr(self, "_last_sources", [])

    # File management

    def save_uploaded_file(self, file_obj, filename: str) -> str:
        """
        Save a Streamlit UploadedFile to uploads_dir.
        Returns the full path to the saved file.
        """
        dest = os.path.join(self.uploads_dir, filename)
        with open(dest, "wb") as f:
            f.write(file_obj.read())
        return dest

    def delete_file(self, filename: str) -> bool:
        """
        Remove a file from disk AND remove its chunks from the index.
        Returns True on success.
        """
      
        self.vector_store.remove_file(filename)
        self.vector_store.save()

        path = os.path.join(self.uploads_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def get_indexed_files(self) -> List[str]:
        """Return all filenames currently in the vector index."""
        return self.vector_store.get_indexed_files()

    def get_uploaded_files(self) -> List[Dict]:
        """
        Return metadata for all files in uploads_dir.
        Each dict has: name, size_bytes, extension, is_indexed.
        """
        indexed = set(self.get_indexed_files())
        results = []
        for fname in os.listdir(self.uploads_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".pdf", ".txt"):
                continue
            fpath = os.path.join(self.uploads_dir, fname)
            results.append({
                "name":       fname,
                "size_bytes": os.path.getsize(fpath),
                "extension":  ext,
                "is_indexed": fname in indexed,
                "path":       fpath,
            })
        return results

    def chunk_count(self) -> int:
        return self.vector_store.chunk_count()

    # Persistence 

    def load(self) -> bool:
        """Load an existing vector store from disk. Returns True if found."""
        return self.vector_store.load()

    def save(self):
        """Persist the current vector store to disk."""
        self.vector_store.save()

    # Helpers

    def _get_all_upload_paths(self) -> List[str]:
        """All valid PDF/TXT paths in uploads_dir."""
        paths = []
        for fname in os.listdir(self.uploads_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".pdf", ".txt"):
                paths.append(os.path.join(self.uploads_dir, fname))
        return paths

    def get_llm_status(self) -> Dict:
        """Check if Ollama is running and the model is available."""
        running = self.llm.is_available()
        model_ready = self.llm.is_model_pulled() if running else False
        models = self.llm.list_models() if running else []
        return {
            "running":     running,
            "model_ready": model_ready,
            "model":       self.llm.model,
            "models":      models,
        }
