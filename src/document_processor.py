"""
src/document_processor.py
─────────────────────────
Handles loading, parsing, and chunking of PDF and TXT documents.
Uses PyMuPDF (fitz) for PDF extraction and plain read() for TXT files.

Chunking strategy — sized in TOKENS, using the embedding model's own tokenizer:

  - Chunk size: 220 tokens
  - Overlap:    32 tokens

Why tokens and not words: the encoder (all-MiniLM-L6-v2) has a hard ceiling of
256 tokens and SILENTLY discards everything past it. Chunking at 512 *words*
produced ~875-token chunks, of which only the first 254 were ever embedded —
so two thirds of every document sat in the index, unreachable by search.

220 leaves headroom under the 254-token usable budget (256 minus [CLS]/[SEP]).
Chunk boundaries snap to word starts, and every chunk is verified to re-encode
within the limit before it is emitted.
"""

import os
import re
import hashlib
import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)

# Used only when DocumentProcessor is constructed without a tokenizer provider
# (standalone use). RAGPipeline always injects the live encoder's tokenizer.
_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Data model

@dataclass
class DocumentChunk:
  
    text: str                   
    source_file: str
    chunk_index: int            
    total_chunks: int           
    page_number: Optional[int] = None
    char_start: int = 0

    # Identity of THIS CHUNK, not of the document. It is a hash of the chunk's
    # own text, so it changes whenever the text changes — which makes it useful
    # for change detection, and useless as a key for replacing a document.
    # (Upsert keys on source_file; see VectorStore.upsert_chunks.)
    chunk_id: str = ""

    def __post_init__(self):
        content_hash = hashlib.md5(
            f"{self.source_file}:{self.chunk_index}:{self.text[:50]}".encode()
        ).hexdigest()[:12]
        self.chunk_id = f"{os.path.splitext(self.source_file)[0]}_{content_hash}"

# Core processor

class DocumentProcessor:
    """
    Loads PDF or TXT files, extracts clean text, and splits it
    into overlapping chunks ready for embedding.
    """

    def __init__(
        self,
        chunk_tokens: int = 220,
        chunk_overlap_tokens: int = 32,
        tokenizer_provider: Optional[Callable] = None,
        max_seq_provider: Optional[Callable] = None,
    ):
        """
        Args:
            chunk_tokens:         Target tokens per chunk, measured with the
                                  EMBEDDING model's tokenizer.
            chunk_overlap_tokens: Tokens repeated at chunk boundaries.
            tokenizer_provider:   Zero-arg callable returning the embedding
                                  model's tokenizer. A callable (not the
                                  tokenizer itself) so the model stays lazily
                                  loaded — nothing is loaded at construction.
            max_seq_provider:     Zero-arg callable returning the encoder's hard
                                  token ceiling.
        """
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self._tokenizer_provider = tokenizer_provider
        self._max_seq_provider = max_seq_provider
        self._tokenizer = None
        self._max_seq = None

        if chunk_overlap_tokens >= chunk_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({chunk_overlap_tokens}) must be smaller "
                f"than chunk_tokens ({chunk_tokens}) — otherwise the window never advances."
            )

    # Tokenizer access (lazy)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            if self._tokenizer_provider is not None:
                self._tokenizer = self._tokenizer_provider()
            else:
                from transformers import AutoTokenizer
                logger.info(f"Loading tokenizer '{_DEFAULT_EMBEDDING_MODEL}' for chunking ...")
                self._tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_EMBEDDING_MODEL)

            if not getattr(self._tokenizer, "is_fast", False):
                raise RuntimeError(
                    "Chunking needs a fast tokenizer (for offset mapping). "
                    f"Got a slow tokenizer for '{_DEFAULT_EMBEDDING_MODEL}'."
                )
        return self._tokenizer

    def _get_max_seq(self) -> int:
        if self._max_seq is None:
            self._max_seq = self._max_seq_provider() if self._max_seq_provider else 256
        return self._max_seq

    def _content_budget(self) -> int:
        """Tokens available for actual text, after [CLS]/[SEP] are reserved."""
        tok = self._get_tokenizer()
        return self._get_max_seq() - tok.num_special_tokens_to_add(pair=False)

    # Public interface 

    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single file and return a list of DocumentChunk objects.

        Supports: .pdf, .txt
        Raises: ValueError for unsupported extensions.
                FileNotFoundError if the file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        logger.info(f"Processing '{filename}' (type={ext})")

        if ext == ".pdf":
            pages_text = self._extract_pdf(file_path)
        elif ext == ".txt":
            pages_text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: '{ext}'. Use PDF or TXT.")

        
        chunks = self._chunk_pages(pages_text, filename)

        logger.info(f"  → {len(chunks)} chunks created from '{filename}'")
        return chunks

    def process_files(self, file_paths: List[str]) -> List[DocumentChunk]:
        """Process multiple files and return all chunks combined."""
        all_chunks: List[DocumentChunk] = []
        for path in file_paths:
            try:
                chunks = self.process_file(path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process '{path}': {e}")
                # Continue with other files instead of crashing
        return all_chunks

    # Extraction helpers

    def _extract_pdf(self, file_path: str) -> List[dict]:
        """
        Extract text from each page of a PDF using PyMuPDF.
        Returns a list of {'page': N, 'text': '...'} dicts.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF extraction.\n"
                "Install it with:  pip install pymupdf"
            )

        pages = []
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text = self._clean_text(text)
                if text.strip():
                    pages.append({"page": page_num + 1, "text": text})
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Could not read PDF '{file_path}': {e}")

        if not pages:
            raise ValueError(f"No readable text found in PDF '{file_path}'.")

        return pages

    def _extract_txt(self, file_path: str) -> List[dict]:
        """
        Read a plain text file.
        Returns a list with one entry (no page numbers for TXT).
        """
        encodings = ["utf-8", "latin-1", "cp1252"]
        text = None
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise RuntimeError(f"Could not decode '{file_path}'. Try saving as UTF-8.")

        text = self._clean_text(text)
        if not text.strip():
            raise ValueError(f"File '{file_path}' appears to be empty.")

        return [{"page": None, "text": text}]

    # Text cleaning 

    def _clean_text(self, text: str) -> str:
        """
        Remove junk characters and normalize whitespace.
        PDF extraction often produces lots of extra newlines and spaces.
        """

        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)         
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    # Chunking 

    def _chunk_pages(
        self,
        pages: List[dict],
        filename: str,
    ) -> List[DocumentChunk]:
        """
        Slide a `chunk_tokens`-wide window (with `chunk_overlap_tokens` overlap)
        over the document, measured with the EMBEDDING model's tokenizer.

        Chunk text is sliced out of the ORIGINAL string by character offset —
        not reconstructed via decode() — so it is byte-exact, and char_start is
        a real character position.
        """
        if not pages:
            return []

        tokenizer = self._get_tokenizer()
        budget = self._content_budget()

        if self.chunk_tokens > budget:
            raise ValueError(
                f"chunk_tokens ({self.chunk_tokens}) exceeds the encoder's usable "
                f"budget ({budget} = max_seq_length {self._get_max_seq()} minus special "
                f"tokens). Chunks would be silently truncated at embed time."
            )

        # Flatten pages into one string, remembering where each page begins so a
        # chunk can be attributed back to a page.
        text, page_starts = self._flatten_pages(pages)
        if not text.strip():
            return []

        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,          # suppress the harmless "longer than 256" notice
        )
        offsets = enc["offset_mapping"]
        word_ids = enc.word_ids()
        n_tokens = len(offsets)
        if n_tokens == 0:
            return []

        chunks: List[DocumentChunk] = []
        start = 0

        while start < n_tokens:
            end = min(start + self.chunk_tokens, n_tokens)

            # Snap the tail back to a word boundary so we never cut a word in
            # half mid-wordpiece ("tokenization" -> "token" + "##ization").
            # Never snap the final chunk — that would drop the document's tail.
            if end < n_tokens:
                end = self._snap_to_word_start(word_ids, end, floor=start + 1)

            char_a = offsets[start][0]
            char_b = offsets[end - 1][1]
            chunk_text = text[char_a:char_b].strip()

            if chunk_text:
                chunk_text = self._enforce_budget(chunk_text, tokenizer, budget)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source_file=filename,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    page_number=self._page_at(page_starts, char_a),
                    char_start=char_a,
                ))

            if end >= n_tokens:
                break

            # Step back by the overlap from the ACTUAL (snapped) end, then snap
            # the new start to a word boundary too — otherwise a fixed-size step
            # lands mid-word and the next chunk opens on a word fragment.
            # floor=start+1 guarantees the window always advances.
            nxt = max(start + 1, end - self.chunk_overlap_tokens)
            start = self._snap_to_word_start(word_ids, nxt, floor=start + 1)

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        self._log_token_stats(chunks, tokenizer, filename)
        return chunks

    def _flatten_pages(self, pages: List[dict]):
        """
        Join pages into one string. Returns (text, page_starts) where page_starts
        is [(char_offset, page_number), ...] ascending.
        """
        parts, page_starts, cursor = [], [], 0
        for page_info in pages:
            ptext = page_info["text"]
            page_starts.append((cursor, page_info["page"]))
            parts.append(ptext)
            cursor += len(ptext) + 2   # +2 for the "\n\n" joiner
        return "\n\n".join(parts), page_starts

    @staticmethod
    def _page_at(page_starts, char_pos: int) -> Optional[int]:
        """Page number containing char_pos (the page of the chunk's first char)."""
        page = None
        for start, num in page_starts:
            if start <= char_pos:
                page = num
            else:
                break
        return page

    @staticmethod
    def _snap_to_word_start(word_ids, idx: int, floor: int) -> int:
        """
        Walk `idx` back until it lands on the first token of a word, so the chunk
        boundary falls between words. Gives up at `floor` to guarantee progress.
        """
        i = idx
        while i > floor:
            if word_ids[i] is None or word_ids[i] != word_ids[i - 1]:
                return i
            i -= 1
        return idx

    def _enforce_budget(self, chunk_text: str, tokenizer, budget: int) -> str:
        """
        Guarantee the chunk fits the encoder. Slicing on token boundaries should
        already ensure this, but re-encoding a substring is not always identical
        to the original token run — so verify rather than assume. This is the
        exact assumption whose failure caused the original bug.
        """
        enc = tokenizer(chunk_text, add_special_tokens=False,
                        return_offsets_mapping=True, verbose=False)
        ids = enc["input_ids"]
        if len(ids) <= budget:
            return chunk_text

        cut = enc["offset_mapping"][budget - 1][1]
        logger.warning(
            "Chunk re-encoded to %d tokens (budget %d) after boundary snapping — "
            "trimming tail.", len(ids), budget,
        )
        return chunk_text[:cut].strip()

    def _log_token_stats(self, chunks: List[DocumentChunk], tokenizer, filename: str):
        if not chunks:
            return
        lens = [
            len(tokenizer(c.text, add_special_tokens=True, verbose=False)["input_ids"])
            for c in chunks
        ]
        limit = self._get_max_seq()
        over = sum(1 for n in lens if n > limit)
        logger.info(
            "  '%s': %d chunks — tokens min/median/max = %d/%d/%d (limit %d, over: %d)",
            filename, len(chunks), min(lens), int(statistics.median(lens)),
            max(lens), limit, over,
        )
        if over:
            logger.error(
                "%d chunk(s) from '%s' STILL exceed the encoder limit — they will be "
                "silently truncated at embed time.", over, filename,
            )
