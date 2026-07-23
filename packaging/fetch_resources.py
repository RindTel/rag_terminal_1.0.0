"""
build/fetch_resources.py
────────────────────────
Populate resources/ with everything the packaged app needs to run fully offline:

  resources/
    models/qwen2.5-3b-instruct-q4_k_m.gguf   the LLM (~2.1 GB)
    fastembed/                                ONNX embedder + reranker
    tokenizer/all-MiniLM-L6-v2/tokenizer.json chunking tokenizer

Run once on the build machine BEFORE PyInstaller:

    python packaging/fetch_resources.py

Idempotent — skips anything already present. Requires: huggingface_hub, fastembed,
tokenizers (all in the build environment).
"""

import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES = os.path.join(ROOT, "resources")

GGUF_REPO    = "Qwen/Qwen2.5-3B-Instruct-GGUF"
GGUF_FILE    = "qwen2.5-3b-instruct-q4_k_m.gguf"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def fetch_gguf():
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(RESOURCES, "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, GGUF_FILE)
    if os.path.exists(dest):
        print("[skip] GGUF already present:", dest)
        return
    print(f"[fetch] {GGUF_FILE} from {GGUF_REPO} (~2.1 GB) ...")
    src = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    shutil.copy(src, dest)
    print("[ok]  GGUF ->", dest)


def fetch_fastembed():
    cache = os.path.join(RESOURCES, "fastembed")
    os.makedirs(cache, exist_ok=True)
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    print("[fetch] fastembed embedder into", cache, "...")
    TextEmbedding(model_name=EMBED_MODEL, cache_dir=cache)
    print("[fetch] fastembed reranker ...")
    TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=cache)
    print("[ok]  fastembed models cached under", cache)


def fetch_tokenizer():
    from tokenizers import Tokenizer
    dest_dir = os.path.join(RESOURCES, "tokenizer", "all-MiniLM-L6-v2")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "tokenizer.json")
    if os.path.exists(dest):
        print("[skip] tokenizer already present:", dest)
        return
    print("[fetch] tokenizer.json for", EMBED_MODEL, "...")
    tok = Tokenizer.from_pretrained(EMBED_MODEL)
    # Match the runtime adapter: no padding/truncation, so token counts are exact.
    tok.no_padding()
    tok.no_truncation()
    tok.save(dest)
    print("[ok]  tokenizer ->", dest)


if __name__ == "__main__":
    os.makedirs(RESOURCES, exist_ok=True)
    fetch_gguf()
    fetch_fastembed()
    fetch_tokenizer()
    print("\nAll resources ready under", RESOURCES)
