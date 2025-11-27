"""Tests for reranker LRU cache functionality."""

import unittest
from unittest.mock import MagicMock

import numpy as np


class TestRerankerCache(unittest.TestCase):
    """Tests for RerankerCache class."""

    def test_basic_query_cache(self):
        """Test basic query result caching."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_query_results=10)

        results = [{"content": "doc1", "rerank_score": 0.9}]
        cache.put_query_result("test query", ["id1", "id2"], results)

        cached = cache.get_query_result("test query", ["id1", "id2"])
        self.assertEqual(cached, results)

    def test_query_cache_miss(self):
        """Test query cache miss for different candidates."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_query_results=10)

        results = [{"content": "doc1"}]
        cache.put_query_result("test query", ["id1", "id2"], results)

        # Different candidates should miss
        missed = cache.get_query_result("test query", ["id1", "id3"])
        self.assertIsNone(missed)

    def test_query_cache_order_independent(self):
        """Test that candidate ID order doesn't affect cache key."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_query_results=10)

        results = [{"content": "doc1"}]
        cache.put_query_result("query", ["id2", "id1"], results)

        # Same IDs in different order should hit
        cached = cache.get_query_result("query", ["id1", "id2"])
        self.assertEqual(cached, results)

    def test_lru_eviction_query_cache(self):
        """Test LRU eviction in query cache."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_query_results=3)

        # Add 3 queries
        for i in range(3):
            cache.put_query_result(f"query{i}", [f"id{i}"], [{"i": i}])

        # Access query0 to make it recently used
        cache.get_query_result("query0", ["id0"])

        # Add query3 - should evict query1
        cache.put_query_result("query3", ["id3"], [{"i": 3}])

        self.assertIsNone(cache.get_query_result("query1", ["id1"]))
        self.assertIsNotNone(cache.get_query_result("query0", ["id0"]))
        self.assertIsNotNone(cache.get_query_result("query3", ["id3"]))

    def test_embedding_cache(self):
        """Test document embedding caching."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_embeddings=100)

        emb = np.array([1.0, 2.0, 3.0])
        cache.put_embedding("doc1", emb)

        cached = cache.get_embedding("doc1")
        np.testing.assert_array_equal(cached, emb)

    def test_embedding_batch_operations(self):
        """Test batch embedding operations."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache(max_embeddings=100)

        embeddings = {
            "doc1": np.array([1.0, 2.0]),
            "doc2": np.array([3.0, 4.0]),
        }
        cache.put_embeddings_batch(embeddings)

        found, missing = cache.get_embeddings_batch(["doc1", "doc2", "doc3"])

        self.assertEqual(len(found), 2)
        self.assertEqual(missing, ["doc3"])

    def test_query_embedding_cache(self):
        """Test query embedding caching."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache()

        emb = np.array([1.0, 2.0, 3.0])
        cache.put_query_embedding("test query", emb)

        cached = cache.get_query_embedding("test query")
        np.testing.assert_array_equal(cached, emb)

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache()

        cache.put_query_result("q1", ["id1"], [{}])
        cache.get_query_result("q1", ["id1"])  # Hit
        cache.get_query_result("q2", ["id2"])  # Miss

        cache.put_embedding("doc1", np.array([1.0]))
        cache.get_embedding("doc1")  # Hit
        cache.get_embedding("doc2")  # Miss

        stats = cache.stats

        self.assertEqual(stats["query_hits"], 1)
        self.assertEqual(stats["query_misses"], 1)
        self.assertEqual(stats["embedding_hits"], 1)
        self.assertEqual(stats["embedding_misses"], 1)

    def test_clear_cache(self):
        """Test clearing all caches."""
        from src.cubo.rerank.reranker import RerankerCache

        cache = RerankerCache()

        cache.put_query_result("q1", ["id1"], [{}])
        cache.put_embedding("doc1", np.array([1.0]))
        cache.put_query_embedding("q1", np.array([1.0]))

        cache.clear()

        self.assertIsNone(cache.get_query_result("q1", ["id1"]))
        self.assertIsNone(cache.get_embedding("doc1"))
        self.assertIsNone(cache.get_query_embedding("q1"))


class TestLocalRerankerWithCache(unittest.TestCase):
    """Tests for LocalReranker with caching enabled."""

    def test_rerank_uses_cache(self):
        """Test that repeated rerank calls use cache."""
        from src.cubo.rerank.reranker import LocalReranker, RerankerCache

        # Mock model
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))

        cache = RerankerCache()
        reranker = LocalReranker(mock_model, top_n=10, cache=cache)

        candidates = [
            {"id": "doc1", "content": "hello world"},
            {"id": "doc2", "content": "foo bar"},
        ]

        # First call - should compute
        result1 = reranker.rerank("test query", candidates)
        initial_encode_count = mock_model.encode.call_count

        # Second call - should use cache
        result2 = reranker.rerank("test query", candidates)

        # Encode should not be called again
        self.assertEqual(mock_model.encode.call_count, initial_encode_count)

        # Results should be identical
        self.assertEqual(len(result1), len(result2))

    def test_rerank_cache_disabled(self):
        """Test reranking when cache is disabled."""
        from src.cubo.rerank.reranker import LocalReranker, RerankerCache

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([1.0, 0.0]))

        cache = RerankerCache()
        reranker = LocalReranker(mock_model, top_n=10, cache=cache)
        reranker._cache_enabled = False

        candidates = [{"id": "doc1", "content": "hello"}]

        # Two calls
        reranker.rerank("query", candidates)
        call_count_1 = mock_model.encode.call_count

        reranker.rerank("query", candidates)
        call_count_2 = mock_model.encode.call_count

        # Without cache, encode should be called again
        self.assertGreater(call_count_2, call_count_1)

    def test_embedding_cache_during_scoring(self):
        """Test that document embeddings are cached during scoring."""
        from src.cubo.rerank.reranker import LocalReranker, RerankerCache

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([1.0, 0.0, 0.0]))

        cache = RerankerCache()
        reranker = LocalReranker(mock_model, top_n=10, cache=cache)

        candidates = [
            {"id": "doc1", "content": "hello world"},
        ]

        # Clear query cache to force recompute
        reranker.rerank("query1", candidates)

        # Check embedding was cached
        self.assertIsNotNone(cache.get_embedding("doc1"))

    def test_get_cache_stats(self):
        """Test getting cache statistics from reranker."""
        from src.cubo.rerank.reranker import LocalReranker, RerankerCache

        cache = RerankerCache()
        reranker = LocalReranker(None, cache=cache)

        stats = reranker.get_cache_stats()

        self.assertIn("query_cache_size", stats)
        self.assertIn("embedding_cache_size", stats)

    def test_clear_cache(self):
        """Test clearing cache from reranker."""
        from src.cubo.rerank.reranker import LocalReranker, RerankerCache

        cache = RerankerCache()
        cache.put_embedding("doc1", np.array([1.0]))

        reranker = LocalReranker(None, cache=cache)
        reranker.clear_cache()

        self.assertIsNone(cache.get_embedding("doc1"))


class TestGlobalRerankerCache(unittest.TestCase):
    """Tests for global reranker cache instance."""

    def test_get_reranker_cache_singleton(self):
        """Test that get_reranker_cache returns singleton."""
        from src.cubo.rerank.reranker import get_reranker_cache

        cache1 = get_reranker_cache()
        cache2 = get_reranker_cache()

        self.assertIs(cache1, cache2)

    def test_global_cache_configured_from_config(self):
        """Test that global cache respects config settings."""
        from src.cubo.rerank.reranker import get_reranker_cache

        cache = get_reranker_cache()

        # Should have been configured with some values
        self.assertIsNotNone(cache)
        self.assertGreater(cache._max_query_results, 0)


class TestCandidateIdExtraction(unittest.TestCase):
    """Tests for candidate ID extraction."""

    def test_extract_ids_from_metadata(self):
        """Test extracting IDs from metadata.chunk_id."""
        from src.cubo.rerank.reranker import LocalReranker

        reranker = LocalReranker(None)

        candidates = [
            {"metadata": {"chunk_id": "chunk1"}},
            {"metadata": {"chunk_id": "chunk2"}},
        ]

        ids = reranker._get_candidate_ids(candidates)
        self.assertEqual(ids, ["chunk1", "chunk2"])

    def test_extract_ids_from_id_field(self):
        """Test extracting IDs from id field."""
        from src.cubo.rerank.reranker import LocalReranker

        reranker = LocalReranker(None)

        candidates = [
            {"id": "doc1"},
            {"id": "doc2"},
        ]

        ids = reranker._get_candidate_ids(candidates)
        self.assertEqual(ids, ["doc1", "doc2"])

    def test_extract_ids_fallback_to_index(self):
        """Test falling back to index when no ID available."""
        from src.cubo.rerank.reranker import LocalReranker

        reranker = LocalReranker(None)

        candidates = [
            {"content": "doc1"},
            {"content": "doc2"},
        ]

        ids = reranker._get_candidate_ids(candidates)
        self.assertEqual(ids, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
