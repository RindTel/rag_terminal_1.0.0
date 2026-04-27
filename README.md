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

## Quick Start

```bash
ollama pull qwen2.5:7b
pip install -r requirements.txt
streamlit run app.py