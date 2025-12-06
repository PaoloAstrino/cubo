import pytest
pytest.importorskip("torch")

import time

import numpy as np

from cubo.retrieval.cache import SemanticCache


def make_vec(x):
    v = np.array(x, dtype="float32")
    return v.tolist()


def test_semantic_cache_linear_lookup():
    cache = SemanticCache(
        ttl_seconds=10, similarity_threshold=0.99, max_entries=10, use_index=False
    )
    v = make_vec([1.0, 0.0, 0.0])
    cache.add("q1", v, [{"result": "a"}])
    hit = cache.lookup(v)
    assert hit is not None and hit[0]["result"] == "a"


def test_semantic_cache_faiss_lookup():
    cache = SemanticCache(
        ttl_seconds=10,
        similarity_threshold=0.9,
        max_entries=10,
        use_index=True,
        index_type="hnsw",
        hnsw_m=8,
    )
    v1 = make_vec([1.0, 0.0, 0.0])
    v2 = make_vec([0.99, 0.05, 0.0])
    cache.add("q1", v1, [{"result": "doc1"}])
    # Fall back to index lookup
    hit = cache.lookup(v2)
    assert hit is not None and hit[0]["result"] == "doc1"


def test_semantic_cache_ttl_evict():
    cache = SemanticCache(ttl_seconds=1, similarity_threshold=0.9, max_entries=10, use_index=False)
    v = make_vec([0.0, 1.0])
    cache.add("q2", v, [{"result": "b"}])
    time.sleep(1.1)
    hit = cache.lookup(v)
    assert hit is None


def test_semantic_cache_lru_evict():
    cache = SemanticCache(ttl_seconds=600, similarity_threshold=0.9, max_entries=2, use_index=False)
    v1 = make_vec([1.0, 0.0])
    v2 = make_vec([0.0, 1.0])
    v3 = make_vec([1.0, 1.0])
    cache.add("a", v1, [{"result": "a"}])
    cache.add("b", v2, [{"result": "b"}])
    # access a to mark it as recently used
    assert cache.lookup(v1) is not None
    cache.add("c", v3, [{"result": "c"}])
    # b should be evicted (least recently used)
    assert cache.lookup(v2) is None
    assert cache.lookup(v1) is not None
    assert cache.lookup(v3) is not None
