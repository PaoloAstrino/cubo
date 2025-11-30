"""Storage package exports.

Avoid eager imports that can cause test-time import cycles or missing
dependencies; import only embedding backends by default.
"""

# Optional: metadata manager (safe to import)
from . import metadata_manager  # noqa: F401

# Embedding storage backends
from .embedding_store import (  # noqa: F401
    EmbeddingCache,
    EmbeddingStore,
    InMemoryEmbeddingStore,
    MmapEmbeddingStore,
    ShardedEmbeddingStore,
    create_embedding_store,
)

# Document store is optional; avoid failing imports in contexts that only
# need embedding stores (e.g., unit tests for embedding_store).
try:  # noqa: F401
    from .document_store import DocumentStore, LRUDocumentCache
except Exception:
    # Leave unavailable to prevent module import errors during minimal tests.
    DocumentStore = None  # type: ignore
    LRUDocumentCache = None  # type: ignore
