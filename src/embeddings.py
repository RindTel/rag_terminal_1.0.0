"""
src/embeddings.py
─────────────────
Pluggable embedding + tokenizer backend behind VectorStore, selected by
config.EMBED_BACKEND:

  - "sentence-transformers" : PyTorch. Dev default; matches the committed eval.
  - "fastembed"             : ONNX runtime, no PyTorch — ~2 GB smaller in the
                              packaged app. Same underlying MiniLM model, so
                              retrieval parity is expected (and gated by re-running
                              eval/run_eval.py before ship).

Both expose the same surface: encode(texts) -> float32 (N, dim), get_tokenizer(),
get_max_seq_length().

Why the tokenizer matters here: the chunker (DocumentProcessor) sizes chunks with
the embedder's *fast* tokenizer, using offset mapping. sentence-transformers hands
over a HuggingFace fast tokenizer directly; fastembed does not, so the fastembed
backend wraps a standalone `tokenizers.Tokenizer` in a thin HF-compatible adapter
— no `transformers` dependency, so it survives in the bundle.
"""

import os
import logging
import numpy as np

import config

logger = logging.getLogger(__name__)


# HF-fast-tokenizer adapter over a raw tokenizers.Tokenizer

class _Encoding(dict):
    """Mimics the slice of HF's BatchEncoding that DocumentProcessor reads."""

    def __init__(self, ids, offsets, word_ids):
        super().__init__(input_ids=ids, offset_mapping=offsets)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class _FastTokenizerAdapter:
    """
    Presents the subset of the HF fast-tokenizer API that DocumentProcessor uses
    — __call__(..., return_offsets_mapping=True), .word_ids(),
    num_special_tokens_to_add(), and is_fast — backed by a raw
    tokenizers.Tokenizer. No `transformers` dependency, so it works in the bundle.
    """

    is_fast = True

    def __init__(self, tok):
        self._tok = tok
        # Special tokens added to a single sequence (2 for BERT-family: CLS/SEP).
        self._num_special = len(tok.encode("", add_special_tokens=True).ids)

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False,
                 verbose=False):
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        # tokenizers exposes ids/offsets/word_ids as properties (lists).
        return _Encoding(enc.ids, enc.offsets, enc.word_ids)

    def num_special_tokens_to_add(self, pair=False):
        return self._num_special


def _canonical_model_id(model_name):
    """
    Fully-qualified Hub id. sentence-transformers accepts the bare
    'all-MiniLM-L6-v2', but fastembed and tokenizers.from_pretrained need the
    'sentence-transformers/all-MiniLM-L6-v2' form.
    """
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def _load_standalone_tokenizer(model_name):
    """
    A standalone fast tokenizer for `model_name`, wrapped HF-style.

    Prefers a bundled tokenizer.json (offline, packaged app); falls back to a
    one-time Hub download in dev when the bundle copy isn't present.
    """
    from tokenizers import Tokenizer
    path = config.EMBED_TOKENIZER_PATH
    if path and os.path.exists(path):
        logger.info("Loading chunking tokenizer from bundled '%s'.", path)
        tok = Tokenizer.from_file(path)
    else:
        logger.info("Loading chunking tokenizer for '%s' from the Hub (dev).", model_name)
        tok = Tokenizer.from_pretrained(model_name)

    # CRITICAL: the loaded tokenizer often ships with padding/truncation enabled
    # (e.g. pad every input to 128 tokens). The chunker needs the TRUE token
    # count to size chunks — padding would report 128 tokens for a 3-word string
    # and blow up the content budget. Strip both so encode() returns exact lengths.
    tok.no_padding()
    tok.no_truncation()
    return _FastTokenizerAdapter(tok)


# Backends

class STEmbeddingBackend:
    """sentence-transformers (PyTorch) — behaviour identical to the original."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # CPU on purpose: the local LLM occupies the GPU; the encoder is tiny
            # and fast on CPU, and keeping it off-GPU avoids CUDA OOM.
            logger.info("Loading embedding model '%s' (CPU, sentence-transformers) ...",
                        self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("Embedding model ready.")
        return self._model

    def encode(self, texts):
        embs = self._get().encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return embs.astype("float32")

    def get_tokenizer(self):
        return self._get().tokenizer

    def get_max_seq_length(self):
        return self._get().max_seq_length


class FastEmbedBackend:
    """fastembed (ONNX) — no PyTorch. The packaged app's embedder."""

    def __init__(self, model_name):
        # fastembed + the standalone tokenizer need the fully-qualified Hub id.
        self.model_name = _canonical_model_id(model_name)
        self._model = None
        self._tokenizer = None

    def _get(self):
        if self._model is None:
            from fastembed import TextEmbedding
            logger.info("Loading embedding model '%s' (ONNX, fastembed) ...", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, **fastembed_kwargs())
            logger.info("Embedding model ready.")
        return self._model

    def encode(self, texts):
        model = self._get()
        embs = np.vstack(list(model.embed(list(texts)))).astype("float32")
        # Normalise so the FAISS inner-product index measures cosine exactly, matching
        # the ST path (normalize_embeddings=True). Idempotent if already unit-norm.
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embs / norms).astype("float32")

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _load_standalone_tokenizer(self.model_name)
        return self._tokenizer

    def get_max_seq_length(self):
        return config.EMBED_MAX_SEQ


def fastembed_kwargs():
    """
    Extra kwargs for fastembed model constructors.

    Points fastembed at the bundled ONNX cache when it exists (packaged app →
    offline); returns {} in dev so fastembed uses its default cache. Shared by the
    embedder here and the reranker in hybrid_retriever.
    """
    if os.path.isdir(config.FASTEMBED_CACHE):
        return {"cache_dir": config.FASTEMBED_CACHE}
    return {}


def create_embedding_backend(model_name):
    """Return the embedding backend for config.EMBED_BACKEND."""
    if config.EMBED_BACKEND.lower() == "fastembed":
        return FastEmbedBackend(model_name)
    return STEmbeddingBackend(model_name)
