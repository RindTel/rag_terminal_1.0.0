"""
config.py
─────────
Central configuration for QwenRAG.

Edit the values in this file to customise the app without
hunting through multiple source files.
"""

import os

UPLOADS_DIR     = os.getenv("QWENRAG_UPLOADS_DIR",     "uploads")
VECTORSTORE_DIR = os.getenv("QWENRAG_VECTORSTORE_DIR", "vectorstore")

# Ollama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b")

# LLM generation parameters

LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE",  "0.1"))   # Low = factual
LLM_TOP_P        = float(os.getenv("LLM_TOP_P",        "0.9"))
LLM_NUM_CTX      = int(os.getenv("LLM_NUM_CTX",        "4096"))  # Context window
LLM_NUM_PREDICT  = int(os.getenv("LLM_NUM_PREDICT",    "1024"))  # Max output tokens

# Document chunking

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "512"))   # Words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))    # Overlap words

# Retrieval

TOP_K            = int(os.getenv("TOP_K",            "5"))
SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD","0.25"))  # Min cosine similarity

# Embedding model

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"   # 22M params, 384-dim, ~90 MB download
    # Alternatives:
    # "all-mpnet-base-v2"         — 420 MB, 768-dim, more accurate but slower
    # "paraphrase-MiniLM-L3-v2"  — 17 MB, 384-dim, faster but less accurate
)

# Logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
