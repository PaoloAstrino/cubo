"""
Factory that returns a BM25Store implementation instance by backend name.
"""

from typing import Optional

from cubo.config import config


def get_bm25_store(backend: Optional[str] = None, **kwargs):
    """Return a BM25 store instance for the configured backend.

    Default backend read from `bm25.backend` config key; defaults to 'python' implementation.
    """
    backend = backend or config.get("bm25.backend", "python")
    backend = backend.lower() if isinstance(backend, str) else "python"
    # Default to Python-based BM25 store implementation
    if backend == "python":
        from cubo.retrieval.bm25_python_store import BM25PythonStore

        return BM25PythonStore(**kwargs)
    # If an unknown backend was requested, fallback to Python implementation
    try:
        from cubo.retrieval.bm25_python_store import BM25PythonStore

        return BM25PythonStore(**kwargs)
    except Exception:
        raise RuntimeError("No BM25 backend available: ensure BM25PythonStore is present")
