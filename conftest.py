"""
Root conftest.

Its mere presence at the repo root makes pytest prepend that root to sys.path,
so tests can `from src.rag_pipeline import RAGPipeline` and `import config`
regardless of where pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
