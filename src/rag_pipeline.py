"""
src/rag_pipeline.py
───────────────────
The orchestrator that connects:
  DocumentProcessor  →  VectorStore  →  OllamaClient

This is the single class you call from the UI. It manages the full
RAG lifecycle: ingest documents, retrieve relevant chunks, generate answers.
"""

import os
import re
import logging
import shutil
from typing import List, Dict, Optional, Tuple

import config
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_client import create_llm_client

logger = logging.getLogger(__name__)

# Refusals. Returned WITHOUT calling the LLM, so the model can never answer
# from parametric memory while appearing to cite the user's documents.

NO_DOCUMENTS_MESSAGE = (
    "⚠️ No documents are indexed yet. Please upload and index some files first."
)

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the indexed documents for this question.\n\n"
    "Nothing in the index scored above the relevance threshold, so no answer was "
    "generated — any answer here would come from the model's own memory rather than "
    "your documents. Try rephrasing, or index a document that covers this topic."
)

# Result dataclass

class RAGResult:
    """Holds the answer and supporting evidence for a single query."""

    def __init__(
        self,
        answer: str,
        sources: List[Dict],
        query: str,
        retrieved_count: int,
        grounded: bool = True,
        no_retrieval: bool = False,
    ):
        self.answer          = answer
        self.sources         = sources   # list of chunk metadata dicts
        self.query           = query
        self.retrieved_count = retrieved_count
        self.grounded        = grounded       # False => not derived from documents
        self.no_retrieval    = no_retrieval   # True  => retrieval returned nothing

    def __repr__(self):
        return (f"<RAGResult query={self.query!r} sources={self.retrieved_count} "
                f"grounded={self.grounded}>")

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
        uploads_dir: str  = None,
        vectorstore_dir: str = None,
        llm_model: str    = None,
        chunk_tokens: int = None,
        chunk_overlap_tokens: int = None,
        top_k: int        = None,
        score_threshold: float = None,
        embedding_model: str = None,
    ):
        """
        Args:
            uploads_dir:          Where uploaded files are stored.
            vectorstore_dir:      Where the FAISS index is persisted.
            llm_model:            Ollama model to use for generation.
            chunk_tokens:         Tokens per chunk, measured with the EMBEDDING
                                  model's tokenizer. Must stay under the encoder's
                                  ceiling (256 for MiniLM) or text is silently
                                  dropped at embed time.
            chunk_overlap_tokens: Token overlap between consecutive chunks.
            top_k:                Number of chunks to retrieve per query.
            score_threshold:      Minimum cosine similarity to include a chunk.
            embedding_model:      sentence-transformers model for embeddings.

        Every argument defaults to the matching value in config.py (which in turn
        honours env vars). Passing an argument explicitly overrides config — that
        is how the eval harness pins settings without touching the file.
        """
        self.uploads_dir  = uploads_dir if uploads_dir is not None else config.UPLOADS_DIR
        self.top_k        = top_k if top_k is not None else config.TOP_K
        self.score_threshold = (
            score_threshold if score_threshold is not None else config.SCORE_THRESHOLD
        )

        vectorstore_dir = (
            vectorstore_dir if vectorstore_dir is not None else config.VECTORSTORE_DIR
        )

        os.makedirs(self.uploads_dir, exist_ok=True)

        # VectorStore first: it owns the embedding model, and the chunker must
        # size chunks with THAT model's tokenizer rather than guessing.
        self.vector_store = VectorStore(
            store_dir=vectorstore_dir,
            model_name=embedding_model if embedding_model is not None else config.EMBEDDING_MODEL,
        )
        self.processor = DocumentProcessor(
            chunk_tokens=(
                chunk_tokens if chunk_tokens is not None else config.CHUNK_TOKENS
            ),
            chunk_overlap_tokens=(
                chunk_overlap_tokens if chunk_overlap_tokens is not None
                else config.CHUNK_OVERLAP_TOKENS
            ),
            tokenizer_provider=self.vector_store.get_tokenizer,
            max_seq_provider=self.vector_store.get_max_seq_length,
        )
        # Backend chosen by config.LLM_BACKEND: OllamaClient in dev, LlamaCppClient
        # in the packaged app. The model arg is honoured by the Ollama backend and
        # ignored by llama.cpp (which uses the bundled GGUF path instead).
        self.llm = create_llm_client(
            model=llm_model if llm_model is not None else config.OLLAMA_MODEL,
        )

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
                # upsert, not add: re-indexing a file must REPLACE its chunks,
                # not stack a second copy of them on top of the first.
                added  = self.vector_store.upsert_chunks(chunks)
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
        Useful if settings (chunk_tokens, embedding model) have changed.
        """
        self.vector_store.clear()
        files = self._get_all_upload_paths()
        if not files:
            # clear() only wipes memory. Without this save(), the stale index
            # stays on disk and every document comes back on the next launch.
            logger.warning("No files in uploads directory to reindex — clearing the index.")
            self.vector_store.save()
            return 0, []
        return self.ingest_files(files, progress_callback)

    # Retrieval 

    # Follow-up signals that mean a query can't stand on its own without the
    # conversation. Whole-word pronouns / references, plus follow-up openers.
    _REFERENTIAL = re.compile(
        r"\b(it|its|it's|they|them|their|theirs|that|this|those|these|"
        r"he|him|his|she|her|hers|one|ones)\b",
        re.I,
    )
    _FOLLOWUP_OPENER = re.compile(r"^\s*(what about|how about|what else|and|also|why)\b", re.I)

    def _looks_referential(self, query: str) -> bool:
        """True if the query needs the conversation to be understood."""
        return bool(
            self._REFERENTIAL.search(query)
            or self._FOLLOWUP_OPENER.match(query)
            or len(query.split()) <= 3
        )

    def _maybe_rewrite_query(self, query: str, chat_history) -> str:
        """
        Rewrite a referential follow-up into a standalone query before retrieval.
        No-op unless rewriting is enabled, there IS history, and the query shows a
        follow-up signal — so standalone questions never pay the extra LLM call.
        """
        if not config.QUERY_REWRITE or not chat_history:
            return query
        if not self._looks_referential(query):
            return query
        return self.llm.rewrite_query(query, chat_history)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Return the most relevant chunks for a query.

        If chat_history is given and the query looks like a follow-up, it is first
        rewritten into a standalone query (see _maybe_rewrite_query).

        Returns:
            List of (chunk_dict, score) sorted by relevance.
        """
        k = top_k or self.top_k
        query = self._maybe_rewrite_query(query, chat_history)

        if config.HYBRID_RETRIEVAL:
            return self._get_hybrid().search(query, top_k=k)

        # Pure-vector path — unchanged, and exactly what runs when the flag is off.
        return self.vector_store.search(
            query,
            top_k=k,
            score_threshold=self.score_threshold,
        )

    def _get_hybrid(self):
        """Lazily build the hybrid retriever (defers the reranker load until needed)."""
        if getattr(self, "_hybrid", None) is None:
            from .hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(self.vector_store)
        return self._hybrid

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
        self._last_sources = []
        self._last_no_retrieval = False
        self._last_prompt_build = None

        if self.vector_store.chunk_count() == 0:
            self._last_no_retrieval = True
            logger.warning("NO_RETRIEVAL (empty index) query=%r", question)
            return self._refuse(question, NO_DOCUMENTS_MESSAGE, stream)

        hits = self.retrieve(question, chat_history=chat_history)
        chunks = [chunk for chunk, _ in hits]
        self._last_sources = chunks  # Cache for streaming mode

        # Retrieval found nothing above score_threshold. Do NOT hand the LLM an
        # empty CONTEXT block — it would answer from parametric memory and the
        # user would have no way to tell the answer was ungrounded.
        if not chunks:
            self._last_no_retrieval = True
            logger.warning(
                "NO_RETRIEVAL (no chunk >= score_threshold=%s) query=%r — skipping generation",
                self.score_threshold, question,
            )
            return self._refuse(question, NO_RELEVANT_CONTEXT_MESSAGE, stream)

        build = self.llm.assemble(
            question=question,
            retrieved_chunks=chunks,
            chat_history=chat_history,
        )

        # Cite what the model actually SAW, not everything retrieval returned.
        # If the budgeter dropped a chunk, listing it as a source would credit
        # the answer to evidence that never reached the model.
        self._last_sources = build.used_chunks
        self._last_prompt_build = build

        if stream:
            return self.llm.generate_stream(build.prompt)

        answer = self.llm.generate(build.prompt)
        return RAGResult(
            answer=answer,
            sources=build.used_chunks,
            query=question,
            retrieved_count=len(build.used_chunks),
        )

    def _refuse(self, question: str, message: str, stream: bool):
        """
        Return a refusal without invoking the LLM.

        Shaped to match query()'s return contract so callers need no special
        casing: a generator when streaming, a RAGResult otherwise.
        """
        if stream:
            def _gen():
                yield message
            return _gen()
        return RAGResult(
            answer=message,
            sources=[],
            query=question,
            retrieved_count=0,
            grounded=False,
            no_retrieval=True,
        )

    def get_last_sources(self) -> List[Dict]:
        """Return the chunks used in the most recent streaming query."""
        return getattr(self, "_last_sources", [])

    def last_prompt_build(self):
        """
        The PromptBuild from the most recent query — how many retrieved chunks
        actually fit the token budget, and how many were dropped. Lets the UI
        surface budget-driven truncation instead of hiding it.
        """
        return getattr(self, "_last_prompt_build", None)

    def last_was_no_retrieval(self) -> bool:
        """
        True if the most recent query retrieved nothing and was refused without
        calling the LLM. Lets streaming callers (the UI) flag the turn as ungrounded.
        """
        return getattr(self, "_last_no_retrieval", False)

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
