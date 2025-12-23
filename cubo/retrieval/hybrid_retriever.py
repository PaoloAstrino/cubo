"""
DEPRECATED shim module. Use `src.cubo.retrieval.retriever.HybridRetriever` instead.
This file remains as a compatibility wrapper: it re-exports the canonical implementation
from `retriever.py` and emits a DeprecationWarning on import so callers can migrate.
"""

import warnings

from cubo.retrieval.retriever import HybridRetriever  # re-export canonical implementation

warnings.warn(
    "src.cubo.retrieval.hybrid_retriever is deprecated; import HybridRetriever from cubo.retrieval.retriever instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HybridRetriever"]
# Deprecated shim - the real `HybridRetriever` implementation lives in `cubo.retrieval.retriever`.
