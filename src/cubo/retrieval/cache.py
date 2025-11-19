"""Embedding-aware semantic cache for retriever responses."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.cubo.utils.logger import logger


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


@dataclass
class CacheEntry:
    query: str
    query_embedding: List[float]
    results: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)
    last_access: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_embedding": self.query_embedding,
            "results": self.results,
            "timestamp": self.timestamp,
            "last_access": self.last_access,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(
            query=data.get("query", ""),
            query_embedding=data.get("query_embedding", []),
            results=data.get("results", []),
            timestamp=data.get("timestamp", time.time()),
            last_access=data.get("last_access", 0),
        )


class SemanticCache:
    """Semantic cache with TTL, similarity threshold, optional persistence and optional FAISS/HNSW index.

    If FAISS is available and enabled, the cache maintains an in-memory HNSW (or flat IP) index for fast
    nearest-neighbor lookup of cached query embeddings. On a hit the cache returns stored results without
    re-querying the vector store, saving round-trips and inference.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        similarity_threshold: float = 0.92,
        max_entries: int = 256,
        cache_path: Optional[str] = None,
        use_index: bool = True,
        index_type: str = 'hnsw',
        hnsw_m: int = 16,
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cache_path = Path(cache_path) if cache_path else None
        self._entries: List[CacheEntry] = []
        self._access_counter = 0
        self._use_index = use_index
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._index = None
        self._dimension: Optional[int] = None
        if self.cache_path:
            self._load_from_disk()
        if self._use_index:
            try:
                import faiss  # type: ignore
            except Exception:
                logger.warning("FAISS is not available; falling back to linear scan for SemanticCache")
                self._use_index = False

    def _load_from_disk(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open('r', encoding='utf-8') as fh:
                raw = json.load(fh)
            self._entries = [CacheEntry.from_dict(item) for item in raw]
            logger.info("Loaded %d semantic cache entries", len(self._entries))
            # Rebuild in-memory index based on loaded entries, if enabled
            if self._use_index and self._entries:
                try:
                    self._rebuild_index()
                except Exception as exc:
                    logger.warning("Failed to rebuild semantic cache index from disk: %s", exc)
        except Exception as exc:
            logger.warning("Failed to load semantic cache: %s", exc)
            self._entries = []

    def _flush_to_disk(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = [entry.to_dict() for entry in self._entries]
            with self.cache_path.open('w', encoding='utf-8') as fh:
                json.dump(data, fh)
        except Exception as exc:
            logger.warning("Failed to persist semantic cache: %s", exc)

    def _evict_expired(self) -> None:
        now = time.time()
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if now - entry.timestamp <= self.ttl_seconds]
        if len(self._entries) != before:
            logger.debug("Evicted %d expired semantic cache entries", before - len(self._entries))

    def _evict_lru(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        # Remove least-recently used entries until we fit within max_entries
        self._entries.sort(key=lambda entry: entry.last_access)
        removed_any = False
        while len(self._entries) > self.max_entries:
            removed = self._entries.pop(0)
            removed_any = True
            logger.debug("SemanticCache LRU evicted entry: %s", removed.query)
        if self._use_index and removed_any:
            self._rebuild_index()

    def lookup(self, query_embedding: List[float], n_results: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        if not query_embedding or not self._entries:
            return None
        self._evict_expired()
        q_vec = np.asarray(query_embedding, dtype='float32')
        best_match: Tuple[float, CacheEntry] | None = None
        # Try index-based lookup if configured
        if self._use_index and self._index is not None and self._dimension is not None:
            try:
                import faiss  # type: ignore
                vec = q_vec.reshape(1, -1).astype('float32')
                if vec.shape[1] == self._dimension:
                    # normalize for inner-product cosine similarity
                    vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
                    vec_norm[vec_norm == 0] = 1.0
                    qnorm = vec / vec_norm
                    k = 1
                    distances, indexes = self._index.search(qnorm, k)
                    idx = int(indexes[0][0]) if indexes[0].size and indexes[0][0] != -1 else -1
                    if idx != -1 and idx < len(self._entries):
                        entry = self._entries[idx]
                        entry_vec = np.asarray(entry.query_embedding, dtype='float32')
                        score = _cosine_similarity(qnorm.flatten(), (entry_vec / np.linalg.norm(entry_vec)))
                        if score >= self.similarity_threshold:
                            best_match = (score, entry)
            except Exception:
                # fallback to linear scan on any index failure
                pass

        if best_match is None:
            for entry in self._entries:
                entry_vec = np.asarray(entry.query_embedding, dtype='float32')
                score = _cosine_similarity(q_vec, entry_vec)
                if score >= self.similarity_threshold:
                    if not best_match or score > best_match[0]:
                        best_match = (score, entry)
        if best_match:
            # update last_access and timestamp for LRU & TTL behavior
            self._access_counter += 1
            best_match[1].last_access = self._access_counter
            old_ts = best_match[1].timestamp
            best_match[1].timestamp = time.time()
            logger.debug("SemanticCache hit: %s old_ts=%s new_ts=%s last_access=%s", best_match[1].query, old_ts, best_match[1].timestamp, best_match[1].last_access)
            logger.debug("Semantic cache hit with similarity %.3f", best_match[0])
            # return slice of results if n_results provided
            results = best_match[1].results
            if n_results is not None and isinstance(n_results, int):
                return results[:n_results]
            return results
        return None

    def add(self, query: str, query_embedding: List[float], results: List[Dict[str, Any]]) -> None:
        if not query_embedding or not results:
            return
        entry = CacheEntry(query=query, query_embedding=query_embedding, results=results)
        self._access_counter += 1
        entry.last_access = self._access_counter
        entry.timestamp = time.time()
        self._entries.append(entry)
        # Update dimension from first added vector
        if self._dimension is None:
            try:
                self._dimension = len(query_embedding)
            except Exception:
                self._dimension = None
        # evict expired and old entries, then rebuild index if needed
        self._evict_expired()
        self._evict_lru()
        if self._use_index:
            try:
                self._rebuild_index()
            except Exception as exc:
                logger.warning("Failed to rebuild semantic cache index on add: %s", exc)
        if self.cache_path:
            self._flush_to_disk()

    def clear(self) -> None:
        self._entries.clear()
        if self.cache_path and self.cache_path.exists():
            try:
                os.remove(self.cache_path)
            except OSError:
                pass

    def _rebuild_index(self) -> None:
        """Rebuild the in-memory FAISS/HNSW index from existing entries."""
        if not self._use_index:
            return
        if not self._entries:
            self._index = None
            return
        try:
            import faiss  # type: ignore
            vectors = np.stack([np.asarray(e.query_embedding, dtype='float32') for e in self._entries])
            # normalize vectors for cosine similarity if using inner product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            dim = vectors.shape[1]
            self._dimension = dim
            if self._index_type == 'hnsw':
                idx = faiss.IndexHNSWFlat(dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
                idx.hnsw.efConstruction = max(40, self._hnsw_m * 2)
                idx.hnsw.efSearch = max(50, self._hnsw_m * 2)
            else:
                idx = faiss.IndexFlatIP(dim)
            idx.add(vectors)
            self._index = idx
        except Exception as exc:
            logger.warning("Failed to build FAISS index for semantic cache: %s", exc)
            self._index = None
