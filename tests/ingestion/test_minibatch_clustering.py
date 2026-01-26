"""Tests for MiniBatchKMeans clustering optimization."""

import numpy as np

from cubo.processing.clustering import SemanticClusterer


class TestMiniBatchKMeans:
    """Tests for MiniBatchKMeans clustering optimization."""

    def test_minibatch_produces_clusters(self):
        """Test that MiniBatchKMeans produces valid clusters."""
        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3, use_minibatch=True)

        # Create separable clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 8) + np.array([-5, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2])

        labels, n_clusters = clusterer.cluster_chunks(embeddings, n_clusters=2)

        assert n_clusters == 2
        assert len(labels) == 40
        assert len(np.unique(labels)) == 2

    def test_minibatch_vs_standard_produces_similar_results(self):
        """Test that MiniBatch produces similar results to standard KMeans."""
        np.random.seed(42)

        # Create well-separated clusters
        cluster1 = np.random.randn(30, 8) + np.array([10, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 8) + np.array([-10, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2])

        # Standard KMeans
        clusterer_standard = SemanticClusterer(method="kmeans", use_minibatch=False)
        labels_std, _ = clusterer_standard._cluster_kmeans(embeddings, n_clusters=2)

        # MiniBatch KMeans
        clusterer_mini = SemanticClusterer(method="kmeans", use_minibatch=True)
        labels_mini, _ = clusterer_mini._cluster_kmeans(embeddings, n_clusters=2)

        # Both should produce 2 clusters
        assert len(np.unique(labels_std)) == 2
        assert len(np.unique(labels_mini)) == 2

        # Cluster assignments should be mostly similar (allowing for label swap)
        # Check if clusters align by comparing cluster sizes
        std_sizes = sorted([np.sum(labels_std == 0), np.sum(labels_std == 1)])
        mini_sizes = sorted([np.sum(labels_mini == 0), np.sum(labels_mini == 1)])

        # Should have similar cluster sizes (within 10% difference)
        assert abs(std_sizes[0] - mini_sizes[0]) <= 6  # 10% of 60

    def test_use_minibatch_parameter(self):
        """Test that use_minibatch parameter is respected."""
        clusterer_mini = SemanticClusterer(method="kmeans", use_minibatch=True)
        assert clusterer_mini.use_minibatch is True

        clusterer_std = SemanticClusterer(method="kmeans", use_minibatch=False)
        assert clusterer_std.use_minibatch is False

    def test_auto_detect_uses_minibatch(self):
        """Test that auto-detection uses minibatch when enabled."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 8)

        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3, use_minibatch=True)

        # Should not raise and should produce clusters
        n_clusters = clusterer._auto_detect_clusters(embeddings)
        assert n_clusters >= 1

    def test_minibatch_with_small_dataset(self):
        """Test MiniBatchKMeans with a small dataset."""
        np.random.seed(42)
        embeddings = np.random.randn(15, 8)

        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3, use_minibatch=True)
        labels, n_clusters = clusterer.cluster_chunks(embeddings, n_clusters=3)

        assert n_clusters == 3
        assert len(labels) == 15

    def test_minibatch_batch_size_adjustment(self):
        """Test that batch size is adjusted for small datasets."""
        np.random.seed(42)

        # Small dataset - batch_size should be capped at n_samples
        small_embeddings = np.random.randn(20, 8)

        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3, use_minibatch=True)
        labels, n_clusters = clusterer.cluster_chunks(small_embeddings, n_clusters=2)

        assert len(labels) == 20
        assert n_clusters == 2


class TestClusteringBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_default_method_is_kmeans(self):
        """Test that default clustering method is still kmeans."""
        clusterer = SemanticClusterer()
        assert clusterer.method == "kmeans"

    def test_hdbscan_fallback_still_works(self):
        """Test that HDBSCAN fallback to KMeans still works."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 8)

        clusterer = SemanticClusterer(method="hdbscan", min_cluster_size=3)

        # Should not raise (falls back to KMeans if HDBSCAN not available)
        labels, n_clusters = clusterer.cluster_chunks(embeddings)
        assert n_clusters >= 1

    def test_too_few_chunks_returns_single_cluster(self):
        """Test that too few chunks returns single cluster."""
        np.random.seed(42)
        embeddings = np.random.randn(2, 8)  # Too few

        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3)
        labels, n_clusters = clusterer.cluster_chunks(embeddings)

        assert n_clusters == 1
        assert all(labels == 0)

    def test_cluster_chunks_api_unchanged(self):
        """Test that cluster_chunks API is unchanged."""
        np.random.seed(42)
        embeddings = np.random.randn(30, 8)

        clusterer = SemanticClusterer(method="kmeans", min_cluster_size=3)

        # With explicit n_clusters
        labels1, n1 = clusterer.cluster_chunks(embeddings, n_clusters=3)
        assert n1 == 3

        # Without n_clusters (auto-detect)
        labels2, n2 = clusterer.cluster_chunks(embeddings)
        assert n2 >= 1
