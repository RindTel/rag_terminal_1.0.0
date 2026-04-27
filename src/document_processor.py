"""
src/document_processor.py
─────────────────────────
Handles loading, parsing, and chunking of PDF and TXT documents.
Uses PyMuPDF (fitz) for PDF extraction and plain read() for TXT files.

Chunking strategy:
  - Chunk size: 512 tokens (~400 words) — balanced for Qwen's context window
  - Overlap:    64 tokens  — prevents losing meaning at chunk boundaries
"""

import os
import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Data model

@dataclass
class DocumentChunk:
  
    text: str                   
    source_file: str
    chunk_index: int            
    total_chunks: int           
    page_number: Optional[int] = None  
    char_start: int = 0         
    doc_id: str = ""            

    def __post_init__(self):
        content_hash = hashlib.md5(
            f"{self.source_file}:{self.chunk_index}:{self.text[:50]}".encode()
        ).hexdigest()[:12]
        self.doc_id = f"{os.path.splitext(self.source_file)[0]}_{content_hash}"

# Core processor

class DocumentProcessor:
    """
    Loads PDF or TXT files, extracts clean text, and splits it
    into overlapping chunks ready for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        """
        Args:
            chunk_size:    Target number of words per chunk.
            chunk_overlap: Number of words to repeat at chunk boundaries
                           so that no context is lost.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

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
        Combine all pages into a sequence of word tokens, then
        slide a window of `chunk_size` words with `chunk_overlap` overlap.
        """
        word_page_pairs: List[tuple] = []
        for page_info in pages:
            words = page_info["text"].split()
            page_num = page_info["page"]
            for w in words:
                word_page_pairs.append((w, page_num))

        if not word_page_pairs:
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        total_words = len(word_page_pairs)

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            slice_pairs = word_page_pairs[start:end]

            chunk_words = [p[0] for p in slice_pairs]
            chunk_text = " ".join(chunk_words)

            page_num = slice_pairs[0][1] if slice_pairs else None

            chunks.append(DocumentChunk(
                text=chunk_text,
                source_file=filename,
                chunk_index=len(chunks),
                total_chunks=0,        
                page_number=page_num,
                char_start=start,
            ))

            step = self.chunk_size - self.chunk_overlap
            start += step

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks
