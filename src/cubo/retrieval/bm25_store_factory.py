"""
Factory that returns a BM25Store implementation instance by backend name.
"""
from typing import Optional
from src.cubo.config import config

def get_bm25_store(backend: Optional[str] = None, **kwargs):
    """Return a BM25 store instance for the configured backend.

    Default backend read from `bm25.backend` config key; defaults to 'whoosh' with fallback to Python.
    """
    backend = backend or config.get('bm25.backend', 'whoosh')
    backend = backend.lower() if isinstance(backend, str) else 'whoosh'
    if backend == 'whoosh':
        try:
            from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
            return BM25WhooshStore(**kwargs)
        except Exception:
            # If Whoosh backend not available, fallback to Python
            from src.cubo.retrieval.bm25_python_store import BM25PythonStore
            return BM25PythonStore()
    elif backend == 'python':
        from src.cubo.retrieval.bm25_python_store import BM25PythonStore
        return BM25PythonStore()
    else:
        # Unrecognized backend; default to Whoosh with fallback to Python
        try:
            from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
            return BM25WhooshStore(**kwargs)
        except Exception:
            from src.cubo.retrieval.bm25_python_store import BM25PythonStore
            return BM25PythonStore()
