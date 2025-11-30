"""Semantic clustering for grouping chunks by similarity before scaffold generation.

Resource Optimization:
- Uses MiniBatchKMeans instead of KMeans for memory efficiency
- Processes data in batches to reduce RAM usage
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from cubo.config import config
from cubo.utils.logger import logger


class SemanticClusterer:
    """Groups chunks by semantic similarity using K-Means clustering.

    Resource Optimization:
    - Uses MiniBatchKMeans by default for memory efficiency
    - Can fall back to standard KMeans if needed
    """

    def __init__(
        self,
        method: str = "kmeans",
        min_cluster_size: int = 3,
        max_clusters: int = 50,
        use_minibatch: Optional[bool] = None,
    ):
        """Initialize the semantic clusterer.

        Args:
            method: Clustering method ('kmeans' or 'hdbscan')
            min_cluster_size: Minimum chunks per cluster
            max_clusters: Maximum number of clusters to consider
            use_minibatch: Use MiniBatchKMeans for memory efficiency (default: auto from config)
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters

        # Auto-detect whether to use minibatch based on laptop mode or explicit setting
        if use_minibatch is None:
            self.use_minibatch = config.get("laptop_mode", False) or config.get(
                "clustering.use_minibatch", True
            )
        else:
            self.use_minibatch = use_minibatch

    def cluster_chunks(
        self, embeddings: np.ndarray, n_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """Cluster chunk embeddings by semantic similarity.

        Args:
            embeddings: Array of shape (n_chunks, embedding_dim)
            n_clusters: Number of clusters (auto-detected if None)

        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        if len(embeddings) < self.min_cluster_size:
            logger.warning(
                f"Too few chunks ({len(embeddings)}) for clustering, using single cluster"
            )
            return np.zeros(len(embeddings), dtype=int), 1

        if n_clusters is None:
            n_clusters = self._auto_detect_clusters(embeddings)

        if self.method == "kmeans":
            return self._cluster_kmeans(embeddings, n_clusters)
        elif self.method == "hdbscan":
            return self._cluster_hdbscan(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

    def _auto_detect_clusters(self, embeddings: np.ndarray) -> int:
        """Auto-detect optimal number of clusters using elbow method.

        Args:
            embeddings: Chunk embeddings

        Returns:
            Optimal number of clusters
        """
        n_samples = len(embeddings)

        # Limit search range based on data size
        min_k = 2
        max_k = min(self.max_clusters, n_samples // self.min_cluster_size)

        if max_k < min_k:
            logger.warning(
                f"Not enough samples for clustering (need {self.min_cluster_size * min_k}, got {n_samples})"
            )
            return 1

        # Try different k values and compute silhouette scores
        # Use MiniBatchKMeans for faster auto-detection
        k_range = range(min_k, min(max_k + 1, 11))  # Test up to 10 clusters max
        silhouette_scores = []

        for k in k_range:
            if self.use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=3,  # Fewer inits for speed
                    batch_size=min(256, n_samples),
                )
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)

        # Choose k with highest silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        logger.info(
            f"Auto-detected {best_k} clusters (silhouette score: {max(silhouette_scores):.3f})"
        )

        return best_k

    def _cluster_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, int]:
        """Cluster using K-Means or MiniBatchKMeans algorithm.

        Uses MiniBatchKMeans by default for memory efficiency on large datasets.

        Args:
            embeddings: Chunk embeddings
            n_clusters: Number of clusters

        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        n_samples = len(embeddings)

        if self.use_minibatch:
            # MiniBatchKMeans: ~50% less RAM, faster for large datasets
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=3,  # Fewer inits (speed vs accuracy tradeoff)
                batch_size=min(256, n_samples),  # Process in batches
                max_iter=100,
            )
            logger.info("Using MiniBatchKMeans for memory efficiency")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        labels = kmeans.fit_predict(embeddings)

        logger.info(f"K-Means clustering created {n_clusters} clusters")
        return labels, n_clusters

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """Cluster using HDBSCAN algorithm (density-based).

        Args:
            embeddings: Chunk embeddings

        Returns:
            Tuple of (cluster_labels, n_clusters)
        """
        try:
            import hdbscan
        except ImportError:
            logger.warning("HDBSCAN not available, falling back to K-Means")
            n_clusters = self._auto_detect_clusters(embeddings)
            return self._cluster_kmeans(embeddings, n_clusters)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=1, metric="euclidean"
        )
        labels = clusterer.fit_predict(embeddings)

        # HDBSCAN uses -1 for noise points, reassign them to nearest cluster
        if (labels == -1).any():
            noise_mask = labels == -1
            valid_labels = labels[~noise_mask]
            if len(valid_labels) > 0:
                # Assign noise points to nearest cluster centroid
                from sklearn.metrics.pairwise import euclidean_distances

                cluster_ids = np.unique(valid_labels)
                centroids = np.array(
                    [embeddings[labels == cid].mean(axis=0) for cid in cluster_ids]
                )
                noise_embeddings = embeddings[noise_mask]
                distances = euclidean_distances(noise_embeddings, centroids)
                nearest_clusters = cluster_ids[distances.argmin(axis=1)]
                labels[noise_mask] = nearest_clusters

        n_clusters = len(np.unique(labels))
        logger.info(f"HDBSCAN clustering created {n_clusters} clusters")

        return labels, n_clusters
