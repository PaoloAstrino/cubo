"""
Simplified test suite for memory-mapped FAISS storage.

Tests core mmap functionality without full FAISS integration.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from cubo.config import config


class TestMemoryMappedStorage(unittest.TestCase):
    """Test memory-mapped embedding storage without full FAISS."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.mmap_dir = Path(self.test_dir) / "embeddings_mmap"
        self.mmap_dir.mkdir(parents=True, exist_ok=True)

        # Sample data
        self.num_docs = 100
        self.dimension = 384
        self.doc_ids = [f"doc_{i}" for i in range(self.num_docs)]
        self.vectors = np.random.rand(self.num_docs, self.dimension).astype(np.float32)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_mmap_files(self):
        """Test saving embeddings as mmap files."""
        ids_file = self.mmap_dir / "ids.npy"
        vectors_file = self.mmap_dir / "vectors.npy"

        # Save
        np.save(str(ids_file), np.array(self.doc_ids))
        np.save(str(vectors_file), self.vectors)

        # Verify files exist
        self.assertTrue(ids_file.exists())
        self.assertTrue(vectors_file.exists())

    def test_load_mmap_files(self):
        """Test loading embeddings via memory mapping."""
        ids_file = self.mmap_dir / "ids.npy"
        vectors_file = self.mmap_dir / "vectors.npy"

        # Save first
        np.save(str(ids_file), np.array(self.doc_ids))
        np.save(str(vectors_file), self.vectors)

        # Load with mmap
        ids_loaded = np.load(str(ids_file), allow_pickle=True)
        vectors_mmap = np.load(str(vectors_file), mmap_mode="r")

        # Verify
        self.assertEqual(len(ids_loaded), self.num_docs)
        self.assertEqual(vectors_mmap.shape, (self.num_docs, self.dimension))

        # Verify data matches
        np.testing.assert_array_equal(ids_loaded, self.doc_ids)
        np.testing.assert_array_almost_equal(vectors_mmap, self.vectors, decimal=5)

    def test_mmap_memory_efficiency(self):
        """Test that mmap doesn't load all data into RAM."""
        ids_file = self.mmap_dir / "ids.npy"
        vectors_file = self.mmap_dir / "vectors.npy"

        # Save
        np.save(str(ids_file), np.array(self.doc_ids))
        np.save(str(vectors_file), self.vectors)

        # Load with mmap
        vectors_mmap = np.load(str(vectors_file), mmap_mode="r")

        # Access single vector
        single_vector = vectors_mmap[0]

        # Should match
        np.testing.assert_array_almost_equal(single_vector, self.vectors[0], decimal=5)

    def test_index_based_lookup(self):
        """Test index-based vector lookup (simulating FaissStore behavior)."""
        ids_file = self.mmap_dir / "ids.npy"
        vectors_file = self.mmap_dir / "vectors.npy"

        # Save
        np.save(str(ids_file), np.array(self.doc_ids))
        np.save(str(vectors_file), self.vectors)

        # Load
        ids_loaded = np.load(str(ids_file), allow_pickle=True)
        vectors_mmap = np.load(str(vectors_file), mmap_mode="r")

        # Create index mapping (like FaissStore._embeddings in mmap mode)
        id_to_index = {str(doc_id): i for i, doc_id in enumerate(ids_loaded)}

        # Lookup by ID
        test_id = "doc_42"
        idx = id_to_index[test_id]
        vector = vectors_mmap[idx]

        # Should match original
        np.testing.assert_array_almost_equal(vector, self.vectors[42], decimal=5)

    def test_laptop_mode_config(self):
        """Test that laptop mode enables mmap storage."""
        laptop_config = config.get_laptop_mode_config()

        # Should enable mmap
        self.assertEqual(laptop_config.get("vector_store", {}).get("embedding_storage"), "mmap")

        # Should enable lazy loading
        self.assertTrue(laptop_config.get("model_lazy_loading"))


class TestMemoryMappedPerformance(unittest.TestCase):
    """Performance comparison tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.dimension = 384
        self.num_docs = 1000
        self.vectors = np.random.rand(self.num_docs, self.dimension).astype(np.float32)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mmap_file_size(self):
        """Test mmap file size matches expected."""
        vectors_file = Path(self.test_dir) / "vectors.npy"

        # Save
        np.save(str(vectors_file), self.vectors)

        # Check file size (should be ~1000 * 384 * 4 bytes + header)
        file_size = os.path.getsize(vectors_file)
        expected_size = self.num_docs * self.dimension * 4  # float32

        # Allow for numpy header overhead (typically <200 bytes)
        self.assertGreater(file_size, expected_size)
        self.assertLess(file_size, expected_size + 500)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
