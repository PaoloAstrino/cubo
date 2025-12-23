import pytest

pytest.importorskip("torch")

from cubo.retrieval.cache import SemanticCache
from cubo.retrieval.retriever import DocumentRetriever


def test_retriever_uses_semantic_cache():
    # Create retriever with no model (fallback in-memory collection)
    retriever = DocumentRetriever(model=None)
    semantic_cache = SemanticCache(
        ttl_seconds=600, similarity_threshold=0.9, max_entries=10, use_index=False
    )
    # Set up semantic cache in both places (retriever and executor)
    retriever.semantic_cache = semantic_cache
    if hasattr(retriever, "executor"):
        retriever.executor.semantic_cache = semantic_cache

    # Add a pre-cached result
    embedding = [1.0, 0.0, 0.0]
    cached_results = [
        {"document": "Test doc", "metadata": {"filename": "test.txt"}, "similarity": 0.95}
    ]
    semantic_cache.add("test-query", embedding, cached_results)

    # Query using same embedding and ensure we retrieve cached result
    candidates = retriever._query_collection_for_candidates(
        query_embedding=embedding, initial_top_k=3, query="test-query"
    )
    assert isinstance(candidates, list)
    assert len(candidates) == len(cached_results)
    assert candidates[0]["document"] == "Test doc"
