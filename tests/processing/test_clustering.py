"""Tests for semantic clustering functionality."""

import numpy as np
import pytest

from src.cubo.processing.clustering import SemanticClusterer


def test_semantic_clusterer_basic():
    """Test basic K-Means clustering."""
    # Create synthetic embeddings (3 clear clusters)
    np.random.seed(42)
    cluster1 = np.random.randn(10, 384) + np.array([5, 0] + [0] * 382)
    cluster2 = np.random.randn(10, 384) + np.array([0, 5] + [0] * 382)
    cluster3 = np.random.randn(10, 384) + np.array([-5, -5] + [0] * 382)
    embeddings = np.vstack([cluster1, cluster2, cluster3])

    clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3)
    labels, n_clusters = clusterer.cluster_chunks(embeddings, n_clusters=3)

    assert len(labels) == 30
    assert n_clusters == 3
    assert len(np.unique(labels)) == 3


def test_semantic_clusterer_auto_detect():
    """Test auto-detection of cluster count."""
    np.random.seed(42)
    # Create 2 clear clusters
    cluster1 = np.random.randn(15, 384) + np.array([10, 0] + [0] * 382)
    cluster2 = np.random.randn(15, 384) + np.array([0, 10] + [0] * 382)
    embeddings = np.vstack([cluster1, cluster2])

    clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3)
    labels, n_clusters = clusterer.cluster_chunks(embeddings, n_clusters=None)

    assert len(labels) == 30
    # Should detect 2 clusters (or close to it)
    assert 2 <= n_clusters <= 4


def test_semantic_clusterer_small_dataset():
    """Test clustering with too few samples."""
    embeddings = np.random.randn(2, 384)

    clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3)
    labels, n_clusters = clusterer.cluster_chunks(embeddings)

    # Should fall back to single cluster
    assert n_clusters == 1
    assert np.all(labels == 0)


def test_semantic_clusterer_invalid_method():
    """Test invalid clustering method."""
    clusterer = SemanticClusterer(method="invalid")
    embeddings = np.random.randn(10, 384)

    with pytest.raises(ValueError, match="Unknown clustering method"):
        clusterer.cluster_chunks(embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
