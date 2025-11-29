import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from benchmarks.retrieval.rag_benchmark import GroundTruthLoader
from src.cubo.retrieval.cache import SemanticCache


def test_semantic_cache_metrics_hit_and_miss():
    cache = SemanticCache(ttl_seconds=600, similarity_threshold=0.9, max_entries=10, use_index=False)
    embedding1 = [1.0, 0.0, 0.0]
    embedding2 = [0.0, 1.0, 0.0]
    results = [{"id": "d1", "text": "foo"}]

    # Initially miss
    assert cache.lookup(embedding1) is None
    # Add entry
    cache.add("q1", embedding1, results)
    # Hit
    assert cache.lookup(embedding1) is not None
    # Miss on different vector
    assert cache.lookup(embedding2) is None

    metrics = cache.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] >= 1
    assert 0 <= metrics["hit_rate"] <= 1
