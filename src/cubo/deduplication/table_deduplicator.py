"""Table-level semantic deduplication helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # Optional heavy deps guarded for lightweight installs
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None


class TableDeduplicator:
    """Clusters semantically similar tables and emits consolidated virtual tables."""

    def __init__(
        self,
        embedder,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        reducer_dims: int = 16,
    ) -> None:
        self.embedder = embedder
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.reducer_dims = reducer_dims

    def create_table_signature(self, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
        header = metadata.get("columns", [])
        lines = ["Columns:" + ",".join(map(str, header))]
        for row in rows[:5]:
            rendered = ",".join(str(row.get(col, "")) for col in header)
            lines.append(rendered)
        return "\n".join(lines)

    def embed_tables(self, tables_df: pd.DataFrame) -> np.ndarray:
        signatures = []
        for _, row in tables_df.iterrows():
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            sample_rows = row.get("sample_rows", []) if isinstance(row.get("sample_rows"), list) else []
            signatures.append(self.create_table_signature(metadata, sample_rows))
        embeddings = self.embedder.encode(signatures, convert_to_numpy=True)
        return embeddings.astype("float32")

    def cluster_tables(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.array([], dtype=int)
        data = embeddings
        if umap and embeddings.shape[1] > self.reducer_dims:
            reducer = umap.UMAP(n_components=self.reducer_dims, metric="cosine")
            data = reducer.fit_transform(embeddings)
        elif embeddings.shape[1] > self.reducer_dims:
            data = embeddings[:, : self.reducer_dims]
        if not hdbscan:
            from sklearn.cluster import AgglomerativeClustering

            n_clusters = min(max(2, data.shape[0] // self.min_cluster_size), data.shape[0])
            if n_clusters <= 1:
                return np.zeros(data.shape[0], dtype=int)
            model = AgglomerativeClustering(n_clusters=n_clusters)
            return model.fit_predict(data)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        return clusterer.fit_predict(data)

    def create_virtual_tables(self, tables_df: pd.DataFrame, cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        if len(cluster_labels) != len(tables_df):
            raise ValueError("Label count mismatch with tables dataframe")
        tables_df = tables_df.reset_index(drop=True)
        virtual_tables: List[Dict[str, Any]] = []
        for cluster_id in sorted(set(cluster_labels)):
            if cluster_id == -1:
                continue
            subset = tables_df[cluster_labels == cluster_id]
            if subset.empty:
                continue
            common_cols = self._find_common_columns(subset)
            merged = {
                "cluster_id": int(cluster_id),
                "file_count": int(len(subset)),
                "common_columns": common_cols,
                "source_tables": subset["chunk_id"].tolist() if "chunk_id" in subset else [],
            }
            virtual_tables.append(merged)
        return virtual_tables

    def _find_common_columns(self, subset: pd.DataFrame) -> List[str]:
        if "metadata" not in subset:
            return []
        column_sets = []
        for metadata in subset["metadata"]:
            cols = metadata.get("columns", []) if isinstance(metadata, dict) else []
            column_sets.append(set(cols))
        if not column_sets:
            return []
        common = column_sets[0]
        for cols in column_sets[1:]:
            common = common.intersection(cols)
        return sorted(common)


__all__ = ["TableDeduplicator"]
