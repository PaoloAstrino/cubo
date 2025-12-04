"""Tests for SQLite-backed FaissStore document storage.

These tests verify the resource-optimized document storage that uses
SQLite instead of in-memory dictionaries.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cubo.retrieval.vector_store import DocumentCache, FaissStore


class TestDocumentCache:
    """Tests for the LRU document cache."""

    def test_cache_put_get(self):
        """Test basic cache put and get operations."""
        cache = DocumentCache(max_size=10)
        cache.put("doc1", "Hello world", {"author": "test"})

        result = cache.get("doc1")
        assert result is not None
        assert result["document"] == "Hello world"
        assert result["metadata"] == {"author": "test"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = DocumentCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = DocumentCache(max_size=3)

        # Fill cache
        cache.put("doc1", "Content 1", {})
        cache.put("doc2", "Content 2", {})
        cache.put("doc3", "Content 3", {})

        # Access doc1 to make it recently used
        cache.get("doc1")

        # Add doc4 - should evict doc2 (least recently used)
        cache.put("doc4", "Content 4", {})

        assert cache.get("doc2") is None  # Evicted
        assert cache.get("doc1") is not None  # Still there
        assert cache.get("doc4") is not None  # Just added

    def test_cache_remove(self):
        """Test explicit cache removal."""
        cache = DocumentCache(max_size=10)
        cache.put("doc1", "Content", {})

        cache.remove("doc1")
        assert cache.get("doc1") is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = DocumentCache(max_size=10)
        cache.put("doc1", "Content 1", {})
        cache.put("doc2", "Content 2", {})

        cache.clear()
        assert cache.get("doc1") is None
        assert cache.get("doc2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = DocumentCache(max_size=10)

        cache.put("doc1", "Content", {})
        cache.get("doc1")  # Hit
        cache.get("doc2")  # Miss

        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0


class TestFaissStoreSQLite:
    """Tests for SQLite-backed FaissStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_add_and_count(self, temp_dir):
        """Test adding documents and counting."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2", "doc3"]
        vectors = [np.ones(8) * (i + 1) for i in range(3)]
        docs = ["Document 1", "Document 2", "Document 3"]
        metas = [{"idx": i} for i in range(3)]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        assert store.count() == 3

    def test_get_by_ids(self, temp_dir):
        """Test retrieving documents by IDs."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2"]
        vectors = [np.ones(8), np.ones(8) * 2]
        docs = ["First document", "Second document"]
        metas = [{"type": "a"}, {"type": "b"}]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        result = store.get(ids=["doc1"])
        assert "doc1" in result["ids"]
        assert "First document" in result["documents"][0]

    def test_get_with_where_filter(self, temp_dir):
        """Test retrieving documents with metadata filter."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2", "doc3"]
        vectors = [np.ones(8) * i for i in range(1, 4)]
        docs = ["Doc A", "Doc B", "Doc C"]
        metas = [{"category": "x"}, {"category": "y"}, {"category": "x"}]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        result = store.get(where={"category": "x"})
        assert len(result["ids"]) == 2
        assert "doc1" in result["ids"]
        assert "doc3" in result["ids"]

    def test_sqlite_persistence(self, temp_dir):
        """Test that documents persist in SQLite across store instances."""
        index_dir = temp_dir / "faiss_index"

        # Create store and add documents
        store1 = FaissStore(dimension=8, index_dir=index_dir)
        vectors = [np.ones(8)]
        store1.add(
            embeddings=vectors,
            documents=["Persistent document"],
            metadatas=[{"persistent": True}],
            ids=["persist1"],
        )

        # Create new store instance pointing to same directory
        store2 = FaissStore(dimension=8, index_dir=index_dir)

        # Should be able to retrieve document from SQLite
        result = store2.get(ids=["persist1"])
        assert "persist1" in result["ids"]
        assert "Persistent document" in result["documents"][0]

    def test_delete(self, temp_dir):
        """Test deleting documents."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2", "doc3"]
        vectors = [np.ones(8) * i for i in range(1, 4)]
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        metas = [{} for _ in range(3)]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)
        assert store.count() == 3

        store.delete(ids=["doc2"])
        assert store.count() == 2

        result = store.get(ids=["doc2"])
        assert "doc2" not in result["ids"]

    def test_reset(self, temp_dir):
        """Test resetting the store."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        vectors = [np.ones(8)]
        store.add(embeddings=vectors, documents=["Doc"], metadatas=[{}], ids=["doc1"])
        assert store.count() == 1

        store.reset()
        assert store.count() == 0

    def test_query_uses_cache(self, temp_dir):
        """Test that queries use the document cache."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2"]
        vectors = [np.ones(8), np.ones(8) * 0.9]
        docs = ["Query target", "Similar doc"]
        metas = [{}, {}]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        # First query
        query_vec = np.ones(8)
        result1 = store.query(query_embeddings=[query_vec], n_results=2)

        # Check cache was populated
        stats = store.get_cache_stats()
        assert stats["size"] > 0

    def test_get_cache_stats(self, temp_dir):
        """Test getting cache statistics."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        stats = store.get_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hit_rate" in stats


class TestFaissStoreBackwardCompatibility:
    """Tests to ensure backward compatibility with existing API."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_add_query_flow(self, temp_dir):
        """Test the standard add-then-query flow still works."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        # Add documents
        ids = [f"doc{i}" for i in range(5)]
        vectors = [np.random.randn(8) for _ in range(5)]
        docs = [f"Document {i}" for i in range(5)]
        metas = [{"filename": f"doc{i}.txt"} for i in range(5)]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        # Query
        query_vec = vectors[2]  # Use one of the added vectors
        result = store.query(query_embeddings=[query_vec], n_results=3)

        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result
        assert "ids" in result
        assert len(result["documents"][0]) <= 3

    def test_promote_to_hot_sync(self, temp_dir):
        """Test synchronous promotion still works."""
        store = FaissStore(dimension=8, index_dir=temp_dir / "faiss_index")

        ids = ["doc1", "doc2", "doc3"]
        vectors = [np.ones(8) * (i + 1) for i in range(3)]
        docs = [f"Doc {i}" for i in range(3)]
        metas = [{} for _ in range(3)]

        store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

        # Should not raise
        store.promote_to_hot_sync("doc2")

    def test_save_and_load(self, temp_dir):
        """Test save and load operations."""
        index_dir = temp_dir / "faiss_index"
        store = FaissStore(dimension=8, index_dir=index_dir)

        vectors = [np.ones(8)]
        store.add(embeddings=vectors, documents=["Saved doc"], metadatas=[{}], ids=["saved1"])

        store.save()

        # Create new store and load
        store2 = FaissStore(dimension=8, index_dir=index_dir)
        store2.load()

        # Vectors should be loaded from SQLite
        assert store2.count_vectors() > 0
