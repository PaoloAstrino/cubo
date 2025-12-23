"""Tests for FAISS normalization functionality."""

import json
from pathlib import Path

import numpy as np

from cubo.indexing.faiss_index import FAISSIndexManager


def _random_embeddings(count: int, dimension: int = 768) -> np.ndarray:
    """Generate random embeddings with varying norms."""
    # Create embeddings that are NOT unit vectors
    np.random.seed(42)
    embeddings = np.random.randn(count, dimension).astype("float32")
    # These will have norm ~ sqrt(dimension) ~ 27.7 for dim=768
    return embeddings


def _random_unit_embeddings(count: int, dimension: int = 768) -> np.ndarray:
    """Generate random unit vectors."""
    embeddings = _random_embeddings(count, dimension)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


class TestFAISSNormalization:
    """Test suite for FAISS index normalization."""

    def test_normalize_flag_defaults_to_true(self, tmp_path: Path):
        """Test that normalize flag defaults to True."""
        manager = FAISSIndexManager(dimension=768, index_dir=tmp_path)
        assert manager.normalize is True

    def test_normalize_flag_can_be_disabled(self, tmp_path: Path):
        """Test that normalize flag can be set to False."""
        manager = FAISSIndexManager(dimension=768, index_dir=tmp_path, normalize=False)
        assert manager.normalize is False

    def test_model_path_stored(self, tmp_path: Path):
        """Test that model_path is stored correctly."""
        manager = FAISSIndexManager(
            dimension=768,
            index_dir=tmp_path,
            model_path="./models/test-model",
        )
        assert manager.model_path == "./models/test-model"

    def test_build_normalizes_vectors(self, tmp_path: Path):
        """Test that build_indexes normalizes vectors when normalize=True."""
        embeddings = _random_embeddings(20, dimension=64)
        ids = [f"id_{i}" for i in range(20)]

        # Verify embeddings are NOT unit vectors initially
        norms_before = np.linalg.norm(embeddings, axis=1)
        assert norms_before.mean() > 2.0, "Test embeddings should not be unit vectors"

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            hot_fraction=0.5,
            nlist=4,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings.tolist(), ids)

        # Reconstruct vectors from hot index and verify they're normalized
        hot_vectors = manager.hot_index.reconstruct_n(0, manager.hot_index.ntotal)
        hot_norms = np.linalg.norm(hot_vectors, axis=1)

        np.testing.assert_allclose(
            hot_norms, 1.0, atol=0.001, err_msg="Hot index vectors should be unit vectors"
        )

    def test_build_preserves_vectors_when_not_normalized(self, tmp_path: Path):
        """Test that build_indexes preserves vector norms when normalize=False."""
        embeddings = _random_embeddings(20, dimension=64)
        ids = [f"id_{i}" for i in range(20)]

        original_norms = np.linalg.norm(embeddings, axis=1)

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=False,
            hot_fraction=0.5,
            nlist=4,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings.tolist(), ids)

        # Reconstruct vectors from hot index and verify norms preserved
        hot_vectors = manager.hot_index.reconstruct_n(0, manager.hot_index.ntotal)
        hot_norms = np.linalg.norm(hot_vectors, axis=1)

        # Should be same as original (within floating point tolerance)
        np.testing.assert_allclose(
            hot_norms,
            original_norms[: len(hot_norms)],
            rtol=0.001,
            err_msg="Vectors should preserve original norms when normalize=False",
        )

    def test_search_normalizes_query(self, tmp_path: Path):
        """Test that search normalizes query vector when normalize=True."""
        # Build index with normalized vectors
        embeddings = _random_embeddings(20, dimension=64)
        ids = [f"id_{i}" for i in range(20)]

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            hot_fraction=1.0,  # All in hot
            hnsw_m=8,
        )
        manager.build_indexes(embeddings.tolist(), ids)

        # Search with non-unit query - should still work
        query = embeddings[0].tolist()  # Not normalized
        results = manager.search(query, k=5)

        # Should find itself (id_0) as top result
        assert len(results) > 0
        assert results[0]["id"] == "id_0", "Should find query document as top result"
        assert results[0]["distance"] < 0.1, "Distance to self should be near zero"

    def test_metadata_persistence(self, tmp_path: Path):
        """Test that normalize and model_path are saved and loaded from metadata."""
        embeddings = _random_embeddings(10, dimension=64)
        ids = [f"id_{i}" for i in range(10)]

        # Build and save
        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            model_path="./test/model",
            hot_fraction=0.5,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings.tolist(), ids)
        manager.save()

        # Check metadata file directly
        metadata_path = tmp_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata.get("normalize") is True
        assert metadata.get("model_path") == "./test/model"

        # Load into new manager
        manager2 = FAISSIndexManager(dimension=64, index_dir=tmp_path)
        manager2.load()

        assert manager2.normalize is True
        assert manager2.model_path == "./test/model"

    def test_consistent_search_results_with_normalization(self, tmp_path: Path):
        """Test that normalized index produces correct search results."""
        # Create some related documents
        np.random.seed(123)
        base = np.random.randn(64).astype("float32")

        # Create 5 variations of base (similar) and 5 random (different)
        similar = [base + np.random.randn(64).astype("float32") * 0.1 for _ in range(5)]
        different = [np.random.randn(64).astype("float32") * 10 for _ in range(5)]

        embeddings = similar + different
        ids = [f"similar_{i}" for i in range(5)] + [f"different_{i}" for i in range(5)]

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            hot_fraction=1.0,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings, ids)

        # Search with base vector
        results = manager.search(base.tolist(), k=5)

        # All top results should be from "similar" group
        similar_count = sum(1 for r in results if r["id"].startswith("similar"))
        assert similar_count >= 4, f"Expected at least 4 similar results, got {similar_count}"


class TestNormalizationInteraction:
    """Test interaction between normalization and other features."""

    def test_normalize_with_opq(self, tmp_path: Path):
        """Test that normalization works with OPQ enabled."""
        embeddings = _random_embeddings(200, dimension=64)
        ids = [f"id_{i}" for i in range(200)]

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            use_opq=True,
            opq_m=8,
            hot_fraction=0.3,
            nlist=4,
            m=8,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings.tolist(), ids)

        # Hot vectors should be normalized
        hot_vectors = manager.hot_index.reconstruct_n(0, manager.hot_index.ntotal)
        hot_norms = np.linalg.norm(hot_vectors, axis=1)
        np.testing.assert_allclose(hot_norms, 1.0, atol=0.001)

    def test_normalize_with_append(self, tmp_path: Path):
        """Test that normalization is applied during append operations."""
        # Build initial index
        embeddings1 = _random_embeddings(10, dimension=64)
        ids1 = [f"batch1_{i}" for i in range(10)]

        manager = FAISSIndexManager(
            dimension=64,
            index_dir=tmp_path,
            normalize=True,
            hot_fraction=0.5,
            hnsw_m=8,
        )
        manager.build_indexes(embeddings1.tolist(), ids1)

        # Append more data
        embeddings2 = _random_embeddings(10, dimension=64)
        ids2 = [f"batch2_{i}" for i in range(10)]
        manager.build_indexes(embeddings2.tolist(), ids2, append=True)

        # All vectors should be normalized
        hot_vectors = manager.hot_index.reconstruct_n(0, manager.hot_index.ntotal)
        hot_norms = np.linalg.norm(hot_vectors, axis=1)
        np.testing.assert_allclose(hot_norms, 1.0, atol=0.001)
