"""Retrieval package for dense/sparse search implementations.

This package provides document retrieval functionality including:
- DocumentRetriever: Main retrieval facade
- DocumentStore: Document lifecycle management
- RetrievalExecutor: Core retrieval execution
- DeduplicationManager: Result deduplication
- RetrievalOrchestrator: Tiered retrieval coordination
- RetrievalCacheService: Caching layer
- Pydantic models: Type-safe data structures
"""

from . import bm25_searcher as bm25
from . import retriever

# Export main classes for convenience
from .cache import RetrievalCacheService, SemanticCache
from .document_store import DocumentStore
from .models import ChunkMetadata, RetrievalCandidate, RetrievalResult, ScoreBreakdown
from .orchestrator import (
    DeduplicationManager,
    HybridScorer,
    RetrievalOrchestrator,
    TieredRetrievalManager,
)
from .retrieval_executor import RetrievalExecutor
from .retriever import DocumentRetriever, FaissHybridRetriever, HybridRetriever

__all__ = [
    "bm25",
    "retriever",
    "DocumentRetriever",
    "FaissHybridRetriever",
    "HybridRetriever",
    "DocumentStore",
    "RetrievalExecutor",
    "DeduplicationManager",
    "HybridScorer",
    "RetrievalOrchestrator",
    "TieredRetrievalManager",
    "RetrievalCacheService",
    "SemanticCache",
    "RetrievalCandidate",
    "RetrievalResult",
    "ChunkMetadata",
    "ScoreBreakdown",
]
