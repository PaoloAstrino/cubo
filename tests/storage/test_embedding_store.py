"""Tests for on-disk embedding persistence."""

import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np


class TestEmbeddingCache(unittest.TestCase):
    """Tests for the EmbeddingCache LRU implementation."""

    def test_basic_put_get(self):
        """Test basic put and get operations."""
        from cubo.storage.embedding_store import EmbeddingCache

        cache = EmbeddingCache(max_size=10)
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cache.put("doc1", emb)
        result = cache.get("doc1")

        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, emb)

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        from cubo.storage.embedding_store import EmbeddingCache

        cache = EmbeddingCache(max_size=3)

        # Add 3 items
        for i in range(3):
            cache.put(f"doc{i}", np.array([float(i)]))

        self.assertEqual(len(cache), 3)

        # Access doc0 to make it recently used
        cache.get("doc0")

        # Add doc3, should evict doc1 (least recently used)
        cache.put("doc3", np.array([3.0]))

        self.assertIsNone(cache.get("doc1"))  # Evicted
        self.assertIsNotNone(cache.get("doc0"))  # Still there
        self.assertIsNotNone(cache.get("doc2"))  # Still there
        self.assertIsNotNone(cache.get("doc3"))  # Just added

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        from cubo.storage.embedding_store import EmbeddingCache

        cache = EmbeddingCache(max_size=10)
        cache.put("doc1", np.array([1.0]))

        cache.get("doc1")  # Hit
        cache.get("doc2")  # Miss

        stats = cache.stats
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate_pct"], 50.0)

    def test_batch_operations(self):
        """Test batch get and put."""
        from cubo.storage.embedding_store import EmbeddingCache

        cache = EmbeddingCache(max_size=10)

        embeddings = {f"doc{i}": np.array([float(i)]) for i in range(5)}
        cache.put_batch(embeddings)

        result = cache.get_batch(["doc0", "doc2", "doc4", "missing"])

        self.assertEqual(len(result), 3)
        self.assertIn("doc0", result)
        self.assertIn("doc2", result)
        self.assertIn("doc4", result)
        self.assertNotIn("missing", result)

    def test_cache_always_truthy(self):
        """Test that cache is always truthy (even when empty)."""
        from cubo.storage.embedding_store import EmbeddingCache

        cache = EmbeddingCache(max_size=10)
        self.assertTrue(bool(cache))  # Empty but truthy


class TestInMemoryEmbeddingStore(unittest.TestCase):
    """Tests for in-memory embedding store."""

    def test_basic_operations(self):
        """Test basic add, get, delete operations."""
        from cubo.storage.embedding_store import InMemoryEmbeddingStore

        store = InMemoryEmbeddingStore()
        emb = [1.0, 2.0, 3.0]

        store.add("doc1", emb)
        self.assertEqual(store.count(), 1)

        result = store.get("doc1")
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, emb)

        store.delete("doc1")
        self.assertEqual(store.count(), 0)
        self.assertIsNone(store.get("doc1"))

    def test_batch_operations(self):
        """Test batch add and get."""
        from cubo.storage.embedding_store import InMemoryEmbeddingStore

        store = InMemoryEmbeddingStore()
        embeddings = {f"doc{i}": [float(i)] * 3 for i in range(5)}

        store.add_batch(embeddings)
        self.assertEqual(store.count(), 5)

        result = store.get_batch(["doc0", "doc2", "doc4"])
        self.assertEqual(len(result), 3)

    def test_dtype_float16(self):
        """Test float16 dtype for memory savings."""
        from cubo.storage.embedding_store import InMemoryEmbeddingStore

        store = InMemoryEmbeddingStore(dtype="float16")
        emb = [1.0, 2.0, 3.0]

        store.add("doc1", emb)
        result = store.get("doc1")

        self.assertEqual(result.dtype, np.float16)

    def test_iteration(self):
        """Test keys and items iteration."""
        from cubo.storage.embedding_store import InMemoryEmbeddingStore

        store = InMemoryEmbeddingStore()
        for i in range(3):
            store.add(f"doc{i}", [float(i)])

        keys = list(store.keys())
        self.assertEqual(len(keys), 3)

        items = list(store.items())
        self.assertEqual(len(items), 3)


class TestShardedEmbeddingStore(unittest.TestCase):
    """Tests for sharded on-disk embedding store."""

    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="emb_test_")

    def tearDown(self):
        """Clean up temporary directory."""
        for _ in range(3):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                break
            except Exception:
                time.sleep(0.1)

    def test_basic_operations(self):
        """Test basic add, get, delete with disk persistence."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(
            storage_dir=Path(self.temp_dir), shard_size=5, dtype="float32"
        )

        emb = [1.0, 2.0, 3.0]
        store.add("doc1", emb)

        self.assertEqual(store.count(), 1)

        result = store.get("doc1")
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, emb)

    def test_persistence_across_instances(self):
        """Test that data persists when store is reopened."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        # Create and add data
        store1 = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=5)
        store1.add("doc1", [1.0, 2.0, 3.0])
        store1.add("doc2", [4.0, 5.0, 6.0])

        # Create new instance pointing to same directory
        store2 = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=5)

        self.assertEqual(store2.count(), 2)

        result = store2.get("doc1")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_sharding(self):
        """Test that embeddings are properly sharded."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=3)

        # Add 7 embeddings (should create 3 shards)
        for i in range(7):
            store.add(f"doc{i}", [float(i)] * 4)

        self.assertEqual(store.count(), 7)

        # Check shard files exist
        shard_files = list(Path(self.temp_dir).glob("embeddings_shard_*.npy"))
        self.assertGreater(len(shard_files), 1)

    def test_batch_operations(self):
        """Test batch add and get for efficiency."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=10)

        embeddings = {f"doc{i}": [float(i)] * 4 for i in range(20)}
        store.add_batch(embeddings)

        self.assertEqual(store.count(), 20)

        result = store.get_batch([f"doc{i}" for i in range(0, 20, 5)])
        self.assertEqual(len(result), 4)

    def test_float16_dtype(self):
        """Test float16 storage for memory savings."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(
            storage_dir=Path(self.temp_dir), shard_size=10, dtype="float16"
        )

        store.add("doc1", [1.0, 2.0, 3.0])
        result = store.get("doc1")

        self.assertEqual(result.dtype, np.float16)

    def test_delete_and_compact(self):
        """Test deletion and compaction."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=5)

        for i in range(10):
            store.add(f"doc{i}", [float(i)] * 4)

        # Delete some
        store.delete("doc3")
        store.delete("doc7")

        self.assertEqual(store.count(), 8)
        self.assertIsNone(store.get("doc3"))

        # Compact should clean up
        store.compact()
        self.assertEqual(store.count(), 8)

    def test_cache_stats(self):
        """Test that cache stats are available."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(
            storage_dir=Path(self.temp_dir), shard_size=5, cache_size=100, enable_cache=True
        )

        store.add("doc1", [1.0, 2.0])
        store.get("doc1")  # Should be cache hit
        store.get("doc2")  # Cache miss

        stats = store.cache_stats
        self.assertIsNotNone(stats)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)


class TestCreateEmbeddingStore(unittest.TestCase):
    """Tests for the factory function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="emb_factory_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_memory_store(self):
        """Test creating in-memory store."""
        from cubo.storage.embedding_store import InMemoryEmbeddingStore, create_embedding_store

        store = create_embedding_store(mode="memory")
        self.assertIsInstance(store, InMemoryEmbeddingStore)

    def test_create_sharded_store(self):
        """Test creating sharded store."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore, create_embedding_store

        store = create_embedding_store(mode="npy_sharded", storage_dir=Path(self.temp_dir))
        self.assertIsInstance(store, ShardedEmbeddingStore)

    def test_create_mmap_store(self):
        """Test creating mmap store."""
        from cubo.storage.embedding_store import MmapEmbeddingStore, create_embedding_store

        store = create_embedding_store(mode="mmap", storage_dir=Path(self.temp_dir), dimension=128)
        self.assertIsInstance(store, MmapEmbeddingStore)
        store.close()

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        from cubo.storage.embedding_store import create_embedding_store

        with self.assertRaises(ValueError):
            create_embedding_store(mode="invalid")


class TestMmapEmbeddingStore(unittest.TestCase):
    """Tests for memory-mapped embedding store."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mmap_test_")

    def tearDown(self):
        for _ in range(3):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                break
            except Exception:
                time.sleep(0.1)

    def test_basic_operations(self):
        """Test basic mmap store operations."""
        from cubo.storage.embedding_store import MmapEmbeddingStore

        store = MmapEmbeddingStore(
            storage_path=Path(self.temp_dir) / "embeddings.mmap", dimension=4, max_embeddings=100
        )

        emb = [1.0, 2.0, 3.0, 4.0]
        store.add("doc1", emb)

        result = store.get("doc1")
        np.testing.assert_array_almost_equal(result, emb)

        store.close()

    def test_persistence(self):
        """Test that mmap persists across instances."""
        from cubo.storage.embedding_store import MmapEmbeddingStore

        path = Path(self.temp_dir) / "embeddings.mmap"

        store1 = MmapEmbeddingStore(path, dimension=3, max_embeddings=100)
        store1.add("doc1", [1.0, 2.0, 3.0])
        store1.close()

        store2 = MmapEmbeddingStore(path, dimension=3, max_embeddings=100)
        result = store2.get("doc1")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
        store2.close()

    def test_slot_reuse(self):
        """Test that deleted slots are reused."""
        from cubo.storage.embedding_store import MmapEmbeddingStore

        store = MmapEmbeddingStore(
            storage_path=Path(self.temp_dir) / "embeddings.mmap", dimension=2, max_embeddings=5
        )

        # Fill up slots
        for i in range(5):
            store.add(f"doc{i}", [float(i), float(i)])

        # Delete one
        store.delete("doc2")

        # Add new one - should reuse slot
        store.add("doc_new", [10.0, 10.0])

        self.assertEqual(store.count(), 5)

        store.close()


class TestThreadSafety(unittest.TestCase):
    """Tests for thread safety of embedding stores."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="thread_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_writes(self):
        """Test concurrent writes to sharded store."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=10)

        def writer(start_id: int):
            for i in range(start_id, start_id + 20):
                store.add(f"doc{i}", [float(i)] * 4)

        threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 80 unique documents
        self.assertEqual(store.count(), 80)

    def test_concurrent_reads(self):
        """Test concurrent reads from sharded store."""
        from cubo.storage.embedding_store import ShardedEmbeddingStore

        store = ShardedEmbeddingStore(storage_dir=Path(self.temp_dir), shard_size=10)

        # Add data first
        for i in range(50):
            store.add(f"doc{i}", [float(i)] * 4)

        results = []
        errors = []

        def reader(doc_ids):
            try:
                for did in doc_ids:
                    result = store.get(did)
                    if result is not None:
                        results.append(did)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader, args=([f"doc{i}" for i in range(j, 50, 4)],))
            for j in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)


if __name__ == "__main__":
    unittest.main()
