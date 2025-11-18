"""Simple FAISS index manager for hot/cold retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from src.cubo.utils.logger import logger


class FAISSIndexManager:
    """Manages hot (HNSW) and cold (IVF) FAISS indexes for dense retrieval."""

    def __init__(
        self,
        dimension: int,
        index_dir: Optional[Path] = None,
        nlist: int = 32,
        hnsw_m: int = 16,
        hnsw_ef: int = 32,
        hot_fraction: float = 0.2
    ):
        self.dimension = dimension
        self.index_dir = Path(index_dir or Path.cwd() / "faiss_index")
        self.nlist = nlist
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef
        self.hot_fraction = hot_fraction

        self.hot_index: Optional[faiss.Index] = None
        self.cold_index: Optional[faiss.Index] = None
        self.hot_ids: List[str] = []
        self.cold_ids: List[str] = []

    def build_indexes(self, vectors: List[List[float]], ids: List[str]) -> None:
        if not vectors:
            raise ValueError("Cannot build FAISS indexes without embeddings")
        array = np.asarray(vectors, dtype='float32')
        if len(array) != len(ids):
            raise ValueError("Embeddings and ids must have the same length")

        hot_count = max(1, int(len(ids) * self.hot_fraction))
        hot_count = min(hot_count, len(ids))
        hot_vectors = array[:hot_count]
        cold_vectors = array[hot_count:]
        self.hot_ids = ids[:hot_count]
        self.cold_ids = ids[hot_count:]

        logger.info(f"Building hot index with {hot_vectors.shape[0]} vectors (HNSW M={self.hnsw_m})")
        self.hot_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
        self.hot_index.hnsw.efConstruction = max(40, self.hnsw_m * 2)
        self.hot_index.hnsw.efSearch = self.hnsw_ef
        self.hot_index.add(hot_vectors)

        if cold_vectors.size:
            logger.info(f"Building cold index with {cold_vectors.shape[0]} vectors (nlist={self.nlist})")
            self.cold_index = self._build_cold_index(cold_vectors)
        else:
            logger.info("Skipping cold index because no cold vectors were supplied")

    def _build_cold_index(self, vectors: np.ndarray) -> faiss.Index:
        if vectors.shape[0] <= self.nlist:
            logger.warning("Cold vector count smaller than nlist; falling back to flat index for cold set")
            cold_index = faiss.IndexFlatL2(self.dimension)
            cold_index.add(vectors)
            return cold_index

        quantizer = faiss.IndexFlatL2(self.dimension)
        cold_index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        cold_index.train(vectors)
        cold_index.nprobe = min(self.nlist, 8)
        cold_index.add(vectors)
        return cold_index

    def search(self, query: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not self.hot_index and not self.cold_index:
            raise ValueError("No FAISS indexes built yet")
        query_vec = np.asarray(query, dtype='float32').reshape(1, -1)
        results: List[Dict[str, Any]] = []
        if self.hot_index and self.hot_ids:
            dists, labels = self.hot_index.search(query_vec, k)
            for idx, dist in zip(labels[0], dists[0]):
                if idx == -1:
                    continue
                result = {
                    'id': self.hot_ids[int(idx)],
                    'distance': float(dist),
                    'source': 'hot'
                }
                results.append(result)

        if self.cold_index and self.cold_ids:
            dists, labels = self.cold_index.search(query_vec, k)
            for idx, dist in zip(labels[0], dists[0]):
                if idx == -1:
                    continue
                result = {
                    'id': self.cold_ids[int(idx)],
                    'distance': float(dist),
                    'source': 'cold'
                }
                results.append(result)

        results.sort(key=lambda r: r['distance'])
        return results[:k]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.hot_index:
            faiss.write_index(self.hot_index, str(self.index_dir / 'hot.index'))
        if self.cold_index:
            faiss.write_index(self.cold_index, str(self.index_dir / 'cold.index'))
        metadata = {
            'dimension': self.dimension,
            'hot_ids': self.hot_ids,
            'cold_ids': self.cold_ids,
            'nlist': self.nlist,
            'hnsw_m': self.hnsw_m,
            'hot_fraction': self.hot_fraction
        }
        metadata_path = self.index_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as fh:
            json.dump(metadata, fh, indent=2)
        logger.info(f"Saved FAISS indexes and metadata to {self.index_dir}")

    def load(self) -> None:
        metadata_path = self.index_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError("FAISS metadata not found; run build before load")
        with open(metadata_path, 'r', encoding='utf-8') as fh:
            metadata = json.load(fh)
        self.hot_ids = metadata.get('hot_ids', [])
        self.cold_ids = metadata.get('cold_ids', [])
        self.dimension = metadata.get('dimension', self.dimension)
        hot_path = self.index_dir / 'hot.index'
        cold_path = self.index_dir / 'cold.index'
        if hot_path.exists():
            self.hot_index = faiss.read_index(str(hot_path))
        if cold_path.exists():
            self.cold_index = faiss.read_index(str(cold_path))
