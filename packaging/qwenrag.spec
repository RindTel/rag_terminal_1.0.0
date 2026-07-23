# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for QwenRAG — the double-click desktop bundle.

    python packaging/fetch_resources.py  # populate resources/ first (offline models)
    pyinstaller packaging/qwenrag.spec   # -> dist/QwenRAG/  (onedir)

onedir (not onefile): the bundle carries ~2.5 GB of models; onefile would unpack
that to a temp dir on EVERY launch. onedir launches instantly.

Torch/sentence-transformers/transformers are EXCLUDED on purpose — the frozen app
uses the ONNX fastembed backend (EMBED_BACKEND=fastembed, auto-selected when
frozen), which is ~2 GB smaller. If PyInstaller ever pulls them in transitively,
they are dropped here.
"""

import os
from PyInstaller.utils.hooks import collect_all

# Repo root = parent of this spec's dir (packaging/). SPECPATH is the spec's
# directory, injected by PyInstaller. Absolute so the build works regardless of
# cwd — PyInstaller resolves relative script/data paths against the SPEC dir.
ROOT = os.path.dirname(SPECPATH)

datas, binaries, hiddenimports = [], [], []

# Native/data-heavy packages that PyInstaller can't fully trace by static analysis.
for pkg in ("streamlit", "fastembed", "onnxruntime", "llama_cpp", "tokenizers", "faiss"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# App files. app.py is executed by Streamlit's bootstrap (a data file, not an
# import target), so it must be shipped as data alongside config, theme, src, and
# the bundled models under resources/.
datas += [
    (os.path.join(ROOT, "app.py"), "."),
    (os.path.join(ROOT, "config.py"), "."),
    (os.path.join(ROOT, ".streamlit", "config.toml"), ".streamlit"),
    (os.path.join(ROOT, "src"), "src"),
    (os.path.join(ROOT, "resources"), "resources"),
]

hiddenimports += [
    "streamlit.runtime.scriptrunner.magic_funcs",
    "platformdirs",
]

a = Analysis(
    [os.path.join(ROOT, "desktop.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "sentence_transformers", "transformers"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QwenRAG",
    debug=False,
    strip=False,
    upx=False,
    console=False,   # no terminal window — it's a GUI app
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="QwenRAG",
)
