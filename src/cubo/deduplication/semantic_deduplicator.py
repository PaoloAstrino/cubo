"""Hybrid semantic deduplication pipeline combining MinHash, FAISS ANN, and clustering."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

try:  # Optional imports
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None
    _FAISS_AVAILABLE = False

try:  # Optional density clustering
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None

try:  # Optional reducer before HDBSCAN
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None

try:  # Fallback ANN backend
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover
    hnswlib = None


@dataclass
class DeduplicationResult:
    canonical_map: Dict[str, str]
    cluster_mapping: Dict[str, int]
    representatives: Dict[int, Dict[str, Any]]
    clusters: List[Set[str]]
    metadata: Dict[str, Any]


class HybridDeduplicator:
    """End-to-end semantic deduplication pipeline."""

    def __init__(
        self,
        method: str = "hybrid",
        similarity_threshold: float = 0.92,
        representative_metric: str = "summary_score",
        prefilter: Optional[Dict[str, Any]] = None,
        ann: Optional[Dict[str, Any]] = None,
        clustering: Optional[Dict[str, Any]] = None,
        run_on: str = "scaffold",
    ) -> None:
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.representative_metric = representative_metric
        self.prefilter_cfg = prefilter or {"use_minhash": True, "num_perm": 128, "minhash_threshold": 0.8}
        self.ann_cfg = ann or {"backend": "faiss", "k": 50}
        self.clustering_cfg = clustering or {
            "algorithm": "hdbscan",
            "min_cluster_size": 2,
            "min_samples": 1,
            "umap_dims": 32,
        }
        self.run_on = run_on
        self.similarity_graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        chunks_df: pd.DataFrame,
        embeddings: np.ndarray | Sequence[Sequence[float]] | str,
        output_map_path: Optional[str] = None,
    ) -> DeduplicationResult:
        start_time = time.time()
        emb = self._load_embeddings(embeddings)
        chunk_ids = chunks_df["chunk_id"].tolist()

        self.similarity_graph = self.build_similarity_graph(emb, chunk_ids)
        clusters, cluster_mapping = self.find_clusters(emb)
        representatives = self.select_representatives(clusters, chunks_df)
        canonical_map = self._build_canonical_map(clusters, representatives)

        metadata = {
            "method": self.method,
            "run_on": self.run_on,
            "similarity_threshold": self.similarity_threshold,
            "n_chunks": len(chunk_ids),
            "n_clusters": len(clusters),
            "n_representatives": len(representatives),
            "duration_seconds": round(time.time() - start_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prefilter": self.prefilter_cfg,
            "ann": self.ann_cfg,
            "clustering": self.clustering_cfg,
        }

        result = DeduplicationResult(canonical_map, cluster_mapping, representatives, clusters, metadata)
        if output_map_path:
            self.save_map(output_map_path, result)
        return result

    def build_similarity_graph(
        self,
        embeddings: np.ndarray,
        chunk_ids: Sequence[str],
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(chunk_ids)
        ann_neighbors = self._query_ann_neighbors(embeddings)
        self._add_ann_edges(graph, embeddings, chunk_ids, ann_neighbors)

        if self.prefilter_cfg.get("use_minhash", True):
            candidate_pairs = self._prefilter_minhash(chunk_ids, embeddings)
            self._add_prefilter_edges(graph, embeddings, chunk_ids, candidate_pairs)
        return graph

    def find_clusters(self, embeddings: Optional[np.ndarray] = None) -> Tuple[List[Set[str]], Dict[str, int]]:
        algo = (self.clustering_cfg.get("algorithm") or "graph").lower()
        if algo == "hdbscan" and hdbscan and embeddings is not None:
            return self._cluster_with_hdbscan(embeddings)
        clusters = list(nx.connected_components(self.similarity_graph))
        mapping = {}
        for cid, cluster in enumerate(clusters):
            for node in cluster:
                mapping[node] = cid
        return clusters, mapping

    def select_representatives(
        self,
        clusters: Sequence[Set[str]],
        chunks_df: pd.DataFrame,
    ) -> Dict[int, Dict[str, Any]]:
        if chunks_df.empty:
            return {}
        indexed = chunks_df.set_index("chunk_id", drop=False)
        representatives: Dict[int, Dict[str, Any]] = {}
        for cid, cluster in enumerate(clusters):
            cluster_rows = indexed.loc[list(cluster)] if len(cluster) > 1 else indexed.loc[[next(iter(cluster))]]
            metric_series = self._extract_metric(cluster_rows)
            best_idx = metric_series.idxmax()
            best_row = cluster_rows.loc[best_idx]
            representatives[cid] = {
                "chunk_id": best_row["chunk_id"],
                "score": metric_series.loc[best_idx],
                "cluster_size": len(cluster_rows),
            }
        return representatives

    def save_map(self, path: str, result: DeduplicationResult) -> None:
        payload = {
            "version": "1.0",
            "metadata": result.metadata,
            "canonical_map": result.canonical_map,
            "clusters": {str(i): list(cluster) for i, cluster in enumerate(result.clusters)},
            "representatives": {str(cid): rep for cid, rep in result.representatives.items()},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_embeddings(
        self,
        embeddings: np.ndarray | Sequence[Sequence[float]] | str,
    ) -> np.ndarray:
        if isinstance(embeddings, str):
            return np.load(embeddings)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Embeddings must be 2-D")
        return arr

    def _prefilter_minhash(
        self,
        chunk_ids: Sequence[str],
        embeddings: np.ndarray,
    ) -> Set[Tuple[str, str]]:
        num_perm = int(self.prefilter_cfg.get("num_perm", 128))
        threshold = float(self.prefilter_cfg.get("minhash_threshold", 0.8))
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes: Dict[str, MinHash] = {}
        for idx, chunk_id in enumerate(chunk_ids):
            mh = MinHash(num_perm=num_perm)
            # Use simple tokenization via sign of embedding dims for deterministic hash seeds
            for dim, value in enumerate(embeddings[idx]):
                token = f"{dim}:{value > 0}"
                mh.update(token.encode("utf-8"))
            minhashes[chunk_id] = mh
            lsh.insert(chunk_id, mh)
        pairs: Set[Tuple[str, str]] = set()
        for chunk_id, mh in minhashes.items():
            for candidate in lsh.query(mh):
                if candidate == chunk_id:
                    continue
                pair = tuple(sorted((chunk_id, candidate)))
                pairs.add(pair)
        return pairs

    def _query_ann_neighbors(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        backend = (self.ann_cfg.get("backend") or "faiss").lower()
        k = int(self.ann_cfg.get("k", 50))
        if backend == "faiss":
            return self._faiss_neighbors(embeddings, k)
        if backend == "hnsw" and hnswlib:
            return self._hnsw_neighbors(embeddings, k)
        return self._sklearn_neighbors(embeddings, k)

    def _faiss_neighbors(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not _FAISS_AVAILABLE:
            raise RuntimeError("FAISS backend requested but faiss is not available")
        normalized = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(normalized)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        distances, indices = index.search(normalized, min(k, normalized.shape[0]))
        return distances, indices

    def _hnsw_neighbors(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if hnswlib is None:
            raise RuntimeError("hnswlib backend requested but not installed")
        dim = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=32)
        index.add_items(embeddings)
        index.set_ef(min(400, embeddings.shape[0]))
        labels, distances = index.knn_query(embeddings, k=min(k, embeddings.shape[0]))
        return 1 - distances, labels

    def _sklearn_neighbors(self, embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        normed = embeddings.astype(np.float32)
        nbrs = NearestNeighbors(n_neighbors=min(k, normed.shape[0]), metric="cosine")
        nbrs.fit(normed)
        distances, indices = nbrs.kneighbors(normed)
        return 1 - distances, indices

    def _add_ann_edges(
        self,
        graph: nx.Graph,
        embeddings: np.ndarray,
        chunk_ids: Sequence[str],
        ann_neighbors: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        distances, indices = ann_neighbors
        for idx, node in enumerate(chunk_ids):
            for neighbor_pos in range(1, indices.shape[1]):
                neighbor_idx = int(indices[idx, neighbor_pos])
                if neighbor_idx < 0 or neighbor_idx >= len(chunk_ids):
                    continue
                neighbor_id = chunk_ids[neighbor_idx]
                if neighbor_id == node:
                    continue
                similarity = float(distances[idx, neighbor_pos])
                if similarity >= self.similarity_threshold:
                    graph.add_edge(node, neighbor_id, weight=similarity)

    def _add_prefilter_edges(
        self,
        graph: nx.Graph,
        embeddings: np.ndarray,
        chunk_ids: Sequence[str],
        candidate_pairs: Set[Tuple[str, str]],
    ) -> None:
        idx_lookup = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
        for doc_a, doc_b in candidate_pairs:
            ia = idx_lookup.get(doc_a)
            ib = idx_lookup.get(doc_b)
            if ia is None or ib is None:
                continue
            vec_a = embeddings[ia]
            vec_b = embeddings[ib]
            similarity = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
            if similarity >= self.similarity_threshold:
                graph.add_edge(doc_a, doc_b, weight=similarity)

    def _cluster_with_hdbscan(self, embeddings: np.ndarray) -> Tuple[List[Set[str]], Dict[str, int]]:
        dims = int(self.clustering_cfg.get("umap_dims", 32))
        reduced = embeddings
        if umap and embeddings.shape[1] > dims:
            reducer = umap.UMAP(n_components=dims, metric="cosine", n_neighbors=min(50, len(embeddings) - 1))
            reduced = reducer.fit_transform(embeddings)
        elif embeddings.shape[1] > dims:
            reduced = embeddings[:, :dims]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(self.clustering_cfg.get("min_cluster_size", 2)),
            min_samples=int(self.clustering_cfg.get("min_samples", 1)),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(reduced)
        clusters: Dict[int, Set[str]] = {}
        mapping: Dict[str, int] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            chunk_id = list(self.similarity_graph.nodes())[idx]
            clusters.setdefault(label, set()).add(chunk_id)
            mapping[chunk_id] = label
        # Ensure singletons appear as their own clusters
        for chunk_id in self.similarity_graph.nodes():
            if chunk_id not in mapping:
                new_id = len(clusters)
                clusters[new_id] = {chunk_id}
                mapping[chunk_id] = new_id
        ordered_clusters = [clusters[cid] for cid in sorted(clusters.keys())]
        return ordered_clusters, mapping

    def _extract_metric(self, rows: pd.DataFrame) -> pd.Series:
        metric = self.representative_metric
        if metric == "summary_score" and "summary_score" in rows.columns:
            return rows["summary_score"].fillna(0.0)
        return rows["text"].str.len()

    def _build_canonical_map(
        self,
        clusters: Sequence[Set[str]],
        representatives: Dict[int, Dict[str, Any]],
    ) -> Dict[str, str]:
        canonical_map: Dict[str, str] = {}
        for cid, cluster in enumerate(clusters):
            rep_chunk = representatives.get(cid, {}).get("chunk_id") if representatives else None
            canonical = rep_chunk or next(iter(cluster))
            for chunk_id in cluster:
                canonical_map[chunk_id] = canonical
        return canonical_map


__all__ = ["HybridDeduplicator", "DeduplicationResult"]
