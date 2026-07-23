# Packaging QwenRAG (Windows + Linux)

Goal: a **double-click, zero-setup** app. No Python, no Ollama, no pip, no
internet — everything is bundled.

## Pipeline

1. `python packaging/fetch_resources.py` — downloads the GGUF LLM, the fastembed
   ONNX embedder + reranker, and the chunking tokenizer into `resources/` (~2.5 GB).
2. `pyinstaller --noconfirm packaging/qwenrag.spec` — freezes `desktop.py` + `app.py`
   into `dist/QwenRAG/` (onedir). Torch is excluded; the frozen app runs the ONNX
   path (`EMBED_BACKEND=fastembed`, `LLM_BACKEND=llamacpp`, auto-selected when frozen).
3. Package the onedir per OS (CI does the tar/zip; installers below are the polish step).

Expected artifact size: **~3.2–3.8 GB** per OS (model-dominated).

CI runs all of this: `.github/workflows/build.yml` (Windows + Linux, Python 3.12).

## Installers (polish — not required for a working bundle)

- **Windows** → wrap `dist/QwenRAG/` in an **Inno Setup** or **NSIS** installer for a
  Start-menu shortcut. A plain zip also works (user extracts + runs `QwenRAG.exe`).
- **Linux** → **AppImage** is the double-click equivalent (needs `webkit2gtk` on the
  host for pywebview). A tarball works too: extract + run `./QwenRAG`.

## Code signing — DECISION: ship unsigned (this round)

No certificate yet. Consequences to document for end users:

- **Windows**: SmartScreen shows *"Windows protected your PC"* for an unsigned,
  low-reputation `.exe`. Users click **More info → Run anyway**. Put this in the
  README and on the download page. Revisit an OV (~$200/yr) or EV (~$300–500/yr,
  instant reputation) cert if the audience broadens.
- **Linux**: AppImages aren't signed in practice. Note `chmod +x QwenRAG*.AppImage`.

## Known risk — Streamlit under PyInstaller (timeboxed)

The fiddly part: Streamlit's static assets, package metadata, and script entry
point. `desktop.py` runs the server as a re-exec'd child process (avoids the
"signal only works in main thread" trap); the spec uses `collect_all("streamlit")`.
**If a frozen build won't launch within a few iterations, STOP and report the
specific blocker** rather than guessing — do not burn unlimited build cycles here.
