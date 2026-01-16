"""Top-level package marker for the CUBO codebase."""

# Avoid importing heavy submodules at package import time. Use lazy attribute
# access to fetch components like CuboCore on demand (PEP 562).

__all__ = [
    "CuboCore",
    "api",
    "compression",
    "deduplication",
    "embeddings",
    "evaluation",
    "gui",
    "indexing",
    "ingestion",
    "models",
    "processing",
    "rerank",
    "retrieval",
    "storage",
    "utils",
    "workers",
]


def __getattr__(name: str):
    """Lazily import and expose attributes to avoid heavy imports at module import time."""
    if name == "CuboCore":
        from .core import CuboCore

        return CuboCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
