"""
test_pipeline.py
────────────────
Quick CLI test to verify everything works before running the Streamlit app.

Usage:
    python test_pipeline.py                     # Interactive test menu
    python test_pipeline.py --file myfile.pdf   # Index a specific file and chat
"""

import sys
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import RAGPipeline


def check_ollama(pipeline: RAGPipeline):
    status = pipeline.get_llm_status()
    print("\n── Ollama Status ──────────────────────────")
    if status["running"]:
        print(f"  ✅ Ollama is running at localhost:11434")
        if status["model_ready"]:
            print(f"  ✅ Model '{status['model']}' is available")
        else:
            print(f"  ⚠️  Model '{status['model']}' is NOT pulled yet")
            print(f"      Run:  ollama pull {status['model']}")
    else:
        print("  ❌ Ollama is NOT running")
        print("     Start it with:  ollama serve")
    print(f"  Available models: {status['models'] or '(none)'}")


def check_embeddings(pipeline: RAGPipeline):
    print("\n── Embedding Model ────────────────────────")
    print("  Loading sentence-transformers model (first run downloads ~90 MB) ...")
    try:
        vecs = pipeline.vector_store.embed_texts(["Hello, this is a test sentence."])
        print(f"  ✅ Embeddings OK — shape: {vecs.shape}")
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")


def ingest_file(pipeline: RAGPipeline, path: str):
    print(f"\n── Ingesting '{os.path.basename(path)}' ────────────────")
    added, errors = pipeline.ingest_files([path])
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
    else:
        print(f"  ✅ {added} chunks indexed. Total: {pipeline.chunk_count()}")


def interactive_chat(pipeline: RAGPipeline):
    print("\n── Interactive Chat ────────────────────────")
    print("  Type a question and press Enter. Type 'quit' to exit.\n")
    history = []
    while True:
        try:
            q = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break

        print("  Qwen: ", end="", flush=True)
        stream = pipeline.query(q, chat_history=history, stream=True)
        full = ""
        for token in stream:
            print(token, end="", flush=True)
            full += token
        print()

        sources = pipeline.get_last_sources()
        if sources:
            print(f"\n  Sources: {', '.join(set(s['source_file'] for s in sources))}")

        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant", "content": full})
        print()


def main():
    parser = argparse.ArgumentParser(description="QwenRAG CLI test")
    parser.add_argument("--file", help="Path to a PDF or TXT file to index")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding test")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║         QwenRAG — Pipeline Test          ║")
    print("╚══════════════════════════════════════════╝")

    pipeline = RAGPipeline()
    pipeline.load()

    check_ollama(pipeline)

    if not args.skip_embed:
        check_embeddings(pipeline)

    if args.file:
        if not os.path.exists(args.file):
            print(f"\n  ❌ File not found: {args.file}")
            sys.exit(1)
        dest = os.path.join("uploads", os.path.basename(args.file))
        if args.file != dest:
            import shutil
            os.makedirs("uploads", exist_ok=True)
            shutil.copy2(args.file, dest)
        ingest_file(pipeline, dest)

    files = pipeline.get_indexed_files()
    print(f"\n── Indexed Documents ──────────────────────")
    if files:
        for f in files:
            print(f"  📄 {f}")
        print(f"  Total chunks: {pipeline.chunk_count()}")
    else:
        print("  (no documents indexed yet)")

    status = pipeline.get_llm_status()
    if files and status["running"] and status["model_ready"]:
        interactive_chat(pipeline)
    elif not files:
        print("\n  Tip: Pass --file myfile.pdf to index a document and start chatting.")
    else:
        print("\n  Tip: Fix the Ollama issues above, then run again to chat.")

    print("\n  Done! Run the full app with:  streamlit run app.py\n")

if __name__ == "__main__":
    main()
