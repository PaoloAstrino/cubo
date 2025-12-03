"""Simple FAISS index manager for hot/cold retrieval."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from cubo.indexing.index_publisher import get_current_index_dir
from cubo.utils.logger import logger


class FAISSIndexManager:
    """Manages hot (HNSW) and cold (IVF+PQ) FAISS indexes for dense retrieval."""

    def __init__(
        self,
        dimension: int,
        index_dir: Optional[Path] = None,
        index_root: Optional[Path] = None,
        nlist: int = 32,
        m: int = 8,  # Number of sub-quantizers for PQ
        hnsw_m: int = 16,
        hnsw_ef: int = 32,
        hot_fraction: float = 0.2,
        use_opq: bool = False,
        opq_m: int = 32,
        normalize: bool = True,
        model_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_dir = Path(index_dir or Path.cwd() / "faiss_index")
        self.index_root = Path(index_root) if index_root is not None else None
        self.nlist = nlist
        self.m = m
        self.hnsw_m = hnsw_m
        self.hnsw_ef = hnsw_ef
        self.hot_fraction = hot_fraction
        self.use_opq = use_opq
        self.opq_m = opq_m
        self.normalize = normalize
        self.model_path = model_path

        self.hot_index: Optional[faiss.Index] = None
        self.cold_index: Optional[faiss.Index] = None
        self.hot_ids: List[str] = []
        self.cold_ids: List[str] = []

        self._lock = threading.Lock()

    def build_indexes(
        self, vectors: List[List[float]], ids: List[str], append: bool = False
    ) -> None:
        if not vectors:
            raise ValueError("Cannot build FAISS indexes without embeddings")
        array = np.asarray(vectors, dtype="float32")
        if len(array) != len(ids):
            raise ValueError("Embeddings and ids must have the same length")

        if append and self.hot_index is not None:
            # Append to existing indexes by rebuilding with all data
            logger.info("Appending data to existing indexes by rebuilding...")
            existing_vectors = []
            existing_ids = []

            # Get vectors from hot index
            if self.hot_index and self.hot_ids:
                hot_vectors = self.hot_index.reconstruct_n(0, self.hot_index.ntotal)
                existing_vectors.append(hot_vectors)
                existing_ids.extend(self.hot_ids)

            # Get vectors from cold index if it exists
            if self.cold_index is not None and self.cold_ids:
                cold_vectors = self.cold_index.reconstruct_n(0, self.cold_index.ntotal)
                existing_vectors.append(cold_vectors)
                existing_ids.extend(self.cold_ids)

            if existing_vectors:
                all_vectors = np.vstack(existing_vectors + [array])
                all_ids = existing_ids + ids
            else:
                all_vectors = array
                all_ids = ids
            self._build(all_vectors, all_ids)
            try:
                trace_collector.record(
                    "",
                    "faiss",
                    "faiss.appended",
                    {"hot_count": len(self.hot_ids), "cold_count": len(self.cold_ids)},
                )
            except Exception:
                pass
        else:
            self._build(array, ids)
            try:
                trace_collector.record(
                    "",
                    "faiss",
                    "faiss.built",
                    {"hot_count": len(self.hot_ids), "cold_count": len(self.cold_ids)},
                )
            except Exception:
                pass

    def _build(self, array: np.ndarray, ids: List[str]):
        # Normalize vectors if configured (ensures L2 distance == 2*(1-cosine))
        if self.normalize:
            faiss.normalize_L2(array)
            logger.info("Normalized all vectors to unit length")

        hot_count = max(1, int(len(ids) * self.hot_fraction))
        hot_count = min(hot_count, len(ids))
        hot_vectors = array[:hot_count]
        cold_vectors = array[hot_count:]
        self.hot_ids = ids[:hot_count]
        self.cold_ids = ids[hot_count:]

        logger.info(
            f"Building hot index with {hot_vectors.shape[0]} vectors (HNSW M={self.hnsw_m})"
        )
        self.hot_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
        self.hot_index.hnsw.efConstruction = max(40, self.hnsw_m * 2)
        self.hot_index.hnsw.efSearch = self.hnsw_ef
        self.hot_index.add(hot_vectors)

        if cold_vectors.size:
            logger.info(
                f"Building cold index with {cold_vectors.shape[0]} vectors (nlist={self.nlist}, m={self.m})"
            )
            self.cold_index = self._build_cold_index(cold_vectors)
        else:
            logger.info("Skipping cold index because no cold vectors were supplied")

    def _build_cold_index(self, vectors: np.ndarray) -> faiss.Index:
        # Ensure nlist is not larger than the number of training vectors
        effective_nlist = min(self.nlist, max(1, vectors.shape[0] // 2))
        if vectors.shape[0] <= effective_nlist:
            logger.warning(
                "Cold vector count smaller than nlist; falling back to flat index for cold set"
            )
            cold_index = faiss.IndexFlatL2(self.dimension)
            cold_index.add(vectors)
            return cold_index

        quantizer = faiss.IndexFlatL2(self.dimension)

        # Ensure the number of subquantizers (m) divides our dimension
        m_to_use = self.m
        if self.dimension % m_to_use != 0:
            # Find a smaller divisor of the dimension (fall back to 1 if none)
            for candidate in range(min(m_to_use, self.dimension), 0, -1):
                if self.dimension % candidate == 0:
                    m_to_use = candidate
                    break
            logger.info(
                f"Adjusted PQ M value to {m_to_use} to be compatible with dimension {self.dimension}"
            )

        # Determine an appropriate number of bits for PQ (2**nbits clusters per subquantizer)
        import math

        nbits = max(1, min(8, int(math.floor(math.log2(max(2, vectors.shape[0]))))))

        # FAISS requires a certain number of training points relative to nlist.
        # A rule of thumb (used by FAISS warnings) is ~39 training points per centroid.
        MIN_POINTS_PER_CENTROID = 39
        required_training_points = effective_nlist * MIN_POINTS_PER_CENTROID
        if vectors.shape[0] < required_training_points:
            logger.warning(
                f"Not enough training vectors for IVFPQ (required: {required_training_points}, got: {vectors.shape[0]}). Falling back to Flat index to avoid poor PQ training."
            )
            cold_index = faiss.IndexFlatL2(self.dimension)
            cold_index.add(vectors)
            return cold_index

        # If there are too few training vectors for reasonable clustering, fallback to flat
        # Empirically, require roughly 40 training points per centroid to get stable clustering
        required_training = max(1, effective_nlist * 40)
        if vectors.shape[0] < required_training:
            logger.warning(
                f"Not enough vectors ({vectors.shape[0]}) to reliably train an IVF+PQ index (need >= {required_training}); falling back to flat index"
            )
            cold_index = faiss.IndexFlatL2(self.dimension)
            cold_index.add(vectors)
            return cold_index

        if self.use_opq:
            # Apply OPQ (Optimized Product Quantization) transform
            logger.info(f"Building cold index with OPQ (opq_m={self.opq_m})")
            opq_matrix = faiss.OPQMatrix(self.dimension, self.opq_m)
            opq_matrix.train(vectors)
            # Create IVFPQ index with OPQ preprocessing
            cold_index = faiss.IndexPreTransform(
                opq_matrix,
                faiss.IndexIVFPQ(quantizer, self.dimension, effective_nlist, m_to_use, nbits),
            )
        else:
            cold_index = faiss.IndexIVFPQ(
                quantizer, self.dimension, effective_nlist, m_to_use, nbits
            )

        cold_index.train(vectors)
        if hasattr(cold_index, "index"):
            # For IndexPreTransform, set nprobe on the wrapped index
            cold_index.index.nprobe = min(effective_nlist, 8)
        else:
            cold_index.nprobe = min(effective_nlist, 8)
        cold_index.add(vectors)
        return cold_index

    def search(self, query: List[float], k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.hot_index and not self.cold_index:
                raise ValueError("No FAISS indexes built yet")
            query_vec = np.asarray(query, dtype="float32").reshape(1, -1)
            # Normalize query to match indexed vectors
            if self.normalize:
                faiss.normalize_L2(query_vec)
            results: List[Dict[str, Any]] = []
            if self.hot_index and self.hot_ids:
                dists, labels = self.hot_index.search(query_vec, k)
                for idx, dist in zip(labels[0], dists[0]):
                    if idx == -1:
                        continue
                    result = {
                        "id": self.hot_ids[int(idx)],
                        "distance": float(dist),
                        "source": "hot",
                    }
                    results.append(result)

            if self.cold_index and self.cold_ids:
                dists, labels = self.cold_index.search(query_vec, k)
                for idx, dist in zip(labels[0], dists[0]):
                    if idx == -1:
                        continue
                    result = {
                        "id": self.cold_ids[int(idx)],
                        "distance": float(dist),
                        "source": "cold",
                    }
                    results.append(result)

            results.sort(key=lambda r: r["distance"])
            return results[:k]

    def save(self, path: Optional[Path] = None) -> None:
        save_dir = path or self.index_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        # Write index files to temporary paths and atomically replace to avoid readers seeing partial files
        if self.hot_index:
            tmp_hot = save_dir / "hot.index.tmp"
            final_hot = save_dir / "hot.index"
            faiss.write_index(self.hot_index, str(tmp_hot))
            # Ensure data is flushed to disk and then atomically replace
            os.replace(str(tmp_hot), str(final_hot))
        if self.cold_index:
            tmp_cold = save_dir / "cold.index.tmp"
            final_cold = save_dir / "cold.index"
            faiss.write_index(self.cold_index, str(tmp_cold))
            os.replace(str(tmp_cold), str(final_cold))
        metadata = {
            "dimension": self.dimension,
            "hot_ids": self.hot_ids,
            "cold_ids": self.cold_ids,
            "nlist": self.nlist,
            "m": self.m,
            "hnsw_m": self.hnsw_m,
            "hot_fraction": self.hot_fraction,
            "use_opq": self.use_opq,
            "opq_m": self.opq_m,
            "normalize": self.normalize,
            "model_path": self.model_path,
        }
        metadata_path = save_dir / "metadata.json"
        tmp_meta = save_dir / "metadata.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp_meta), str(metadata_path))
        logger.info(f"Saved FAISS indexes and metadata to {save_dir}")
        try:
            trace_collector.record(
                "",
                "faiss",
                "faiss.save",
                {
                    "hot_count": len(self.hot_ids),
                    "cold_count": len(self.cold_ids),
                    "index_dir": str(save_dir),
                },
            )
        except Exception:
            pass
        # NOTE: Writing to the metadata DB should be done by an index publisher which
        # verifies the written artifacts and atomically flips a pointer file. We do not
        # record the index version here to avoid races with pointer flips.

    def load(self, path: Optional[Path] = None) -> None:
        # If a root is configured check for a pointer file; allow explicit path to override
        if path is not None:
            load_dir = path
        elif self.index_root is not None:
            published = get_current_index_dir(self.index_root)
            if published:
                load_dir = published
            else:
                load_dir = self.index_dir
        else:
            load_dir = self.index_dir
        metadata_path = load_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("FAISS metadata not found; run build before load")
        with open(metadata_path, encoding="utf-8") as fh:
            metadata = json.load(fh)

        with self._lock:
            self.hot_ids = metadata.get("hot_ids", [])
            self.cold_ids = metadata.get("cold_ids", [])
            self.dimension = metadata.get("dimension", self.dimension)
            self.m = metadata.get("m", self.m)
            self.use_opq = metadata.get("use_opq", False)
            self.opq_m = metadata.get("opq_m", 32)
            self.normalize = metadata.get("normalize", True)
            self.model_path = metadata.get("model_path", None)
            hot_path = load_dir / "hot.index"
            cold_path = load_dir / "cold.index"
            if hot_path.exists():
                self.hot_index = faiss.read_index(str(hot_path))
            if cold_path.exists():
                self.cold_index = faiss.read_index(str(cold_path))

    def swap_indexes(self, new_index_dir: Path):
        """Atomically swaps the live indexes with new ones."""
        logger.info(f"Swapping indexes with new ones from {new_index_dir}")
        self.load(path=new_index_dir)
        self.index_dir = new_index_dir
        logger.info("Indexes swapped successfully.")
