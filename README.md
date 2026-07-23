# 🧠 Local RAG Chat with Your Documents

A simple local RAG app for chatting with PDFs and text files privately.

Drop in your documents, ask questions, and get answers powered by **Qwen 2.5** running locally through **Ollama**.

No API keys, no monthly fees, no sending files to someone else’s server.

Everything stays on your machine.

---

## What It Does

- Upload PDF or TXT files
- Automatically extracts and chunks text
- Creates local embeddings
- Uses FAISS for fast semantic search
- Answers questions with Qwen 2.5 via Ollama
- Streams responses live
- Shows which chunks were used as sources
- Supports multiple files
- Saves chat history
- Persistent vector index between restarts
- Dark mode UI

---

## Why I Built It

Most document chat tools either:

- require API keys  
- cost money  
- upload your files to the cloud  
- feel bloated  

I wanted something lightweight, private, and easy to run locally.

So this is that.

---

## Download & Run (no setup)

Grab the bundle for your OS, double-click, and a window opens — no Python, no
Ollama, no installs, and it works offline. Everything (a Qwen 2.5 3B model plus
the search models) is included in the download (~3.5 GB).

- **Windows** — download `QwenRAG-windows-x64.zip`, extract, run `QwenRAG.exe`.
  The first time, Windows SmartScreen may say *"Windows protected your PC"* because
  the app isn't code-signed yet. Click **More info → Run anyway**. (It's the same
  app either way — signing is on the roadmap.)
- **Linux** — download `QwenRAG-linux-x86_64.tar.gz`, extract, then:
  ```bash
  chmod +x QwenRAG/QwenRAG && ./QwenRAG/QwenRAG
  ```

Your uploads and search index are stored in your user data dir
(`%APPDATA%\QwenRAG` on Windows, `~/.local/share/QwenRAG` on Linux), not next to
the app — so they survive updates.

---

## Run from source (developers)

The source workflow uses **Ollama** + PyTorch embeddings (the packaged app swaps
these for an in-process llama.cpp engine and ONNX embeddings — see
[`packaging/PACKAGING.md`](packaging/PACKAGING.md)).

```bash
ollama pull qwen2.5:7b
pip install -r requirements.txt
streamlit run app.py
```

Backends are selectable via env vars (defaults shown):

```bash
LLM_BACKEND=ollama              # or 'llamacpp' (needs LLAMACPP_MODEL_PATH → a .gguf)
EMBED_BACKEND=sentence-transformers   # or 'fastembed' (ONNX, no torch)
```

When the app is packaged (frozen), it auto-selects `llamacpp` + `fastembed`.

### Building the desktop bundle

```bash
python packaging/fetch_resources.py     # download bundled models into resources/
pyinstaller --noconfirm packaging/qwenrag.spec
```

CI builds both OS artifacts automatically — see `.github/workflows/build.yml`.