"""
QwenRAG source package.

The three entry points reach this package from different working directories
(`app.py` adds src/, `eval/run_eval.py` adds ../, `test_pipeline.py` adds the
root), so a plain `import config` from inside src/ is not reliably resolvable.
Put the repo root on sys.path once, here, so every module below can just
`import config`.

Pragmatic for a single-app repo. If this ever becomes a distributable package,
config should move to src/config.py instead.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
