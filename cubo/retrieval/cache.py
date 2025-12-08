"""Embedding-aware semantic cache for retriever responses."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cubo.utils.logger import logger


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
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
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
        index_type: str = "hnsw",
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

        # Metrics tracking for benchmarking
        self._hits = 0
        self._misses = 0
        self._hit_latencies: List[float] = []
        self._miss_latencies: List[float] = []

        if self.cache_path:
            self._load_from_disk()
        if self._use_index:
            try:
                pass  # type: ignore
            except Exception:
                logger.warning(
                    "FAISS is not available; falling back to linear scan for SemanticCache"
                )
                self._use_index = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics for benchmarking."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        avg_hit_latency = (
            sum(self._hit_latencies) / len(self._hit_latencies) if self._hit_latencies else 0.0
        )
        avg_miss_latency = (
            sum(self._miss_latencies) / len(self._miss_latencies) if self._miss_latencies else 0.0
        )

        return {
            "total_queries": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "hit_rate_percent": hit_rate * 100,
            "avg_hit_latency_ms": avg_hit_latency,
            "avg_miss_latency_ms": avg_miss_latency,
            "latency_savings_ms": (
                avg_miss_latency - avg_hit_latency if avg_miss_latency > avg_hit_latency else 0.0
            ),
            "entries_count": len(self._entries),
            "max_entries": self.max_entries,
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._hits = 0
        self._misses = 0
        self._hit_latencies = []
        self._miss_latencies = []

    def _load_from_disk(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
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
            with self.cache_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception as exc:
            logger.warning("Failed to persist semantic cache: %s", exc)

    def _evict_expired(self) -> None:
        now = time.time()
        before = len(self._entries)
        self._entries = [
            entry for entry in self._entries if now - entry.timestamp <= self.ttl_seconds
        ]
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

    def _try_index_lookup(self, q_vec: np.ndarray) -> Optional[Tuple[float, CacheEntry]]:
        """
        Attempt to find a cache match using the FAISS/HNSW index.

        Uses approximate nearest neighbor search for fast lookup when
        the index is available. Falls back to None if index lookup fails.

        Args:
            q_vec: Normalized query embedding vector.

        Returns:
            Tuple of (similarity_score, entry) if match found above threshold,
            None otherwise.
        """
        if not self._use_index or self._index is None or self._dimension is None:
            return None

        try:
            vec = q_vec.reshape(1, -1).astype("float32")
            if vec.shape[1] != self._dimension:
                return None

            # Normalize for inner-product cosine similarity
            vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)
            vec_norm[vec_norm == 0] = 1.0
            qnorm = vec / vec_norm

            # Search for nearest neighbor
            k = 1
            distances, indexes = self._index.search(qnorm, k)
            idx = int(indexes[0][0]) if indexes[0].size and indexes[0][0] != -1 else -1

            if idx == -1 or idx >= len(self._entries):
                return None

            # Verify similarity meets threshold
            entry = self._entries[idx]
            entry_vec = np.asarray(entry.query_embedding, dtype="float32")
            entry_norm = np.linalg.norm(entry_vec)
            if entry_norm == 0:
                return None
            score = _cosine_similarity(qnorm.flatten(), entry_vec / entry_norm)

            if score >= self.similarity_threshold:
                return (score, entry)

        except Exception:
            # Fallback to linear scan on any index failure
            pass

        return None

    def _linear_scan_lookup(self, q_vec: np.ndarray) -> Optional[Tuple[float, CacheEntry]]:
        """
        Find best cache match using linear scan over all entries.

        Used as fallback when index is unavailable or when index
        lookup fails to find a match.

        Args:
            q_vec: Query embedding vector.

        Returns:
            Tuple of (similarity_score, entry) for best match above threshold,
            None if no match found.
        """
        best_match: Optional[Tuple[float, CacheEntry]] = None

        for entry in self._entries:
            entry_vec = np.asarray(entry.query_embedding, dtype="float32")
            score = _cosine_similarity(q_vec, entry_vec)

            if score >= self.similarity_threshold:
                if not best_match or score > best_match[0]:
                    best_match = (score, entry)

        return best_match

    def _update_entry_access(self, entry: CacheEntry) -> None:
        """
        Update entry access tracking for LRU eviction and TTL refresh.

        Args:
            entry: Cache entry to update.
        """
        self._access_counter += 1
        entry.last_access = self._access_counter
        entry.timestamp = time.time()

    def lookup(
        self, query_embedding: List[float], n_results: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Look up cached results for a semantically similar query.

        Searches the cache for an entry whose query embedding has
        cosine similarity above the threshold. Uses FAISS index for
        fast lookup when available, falling back to linear scan.

        Args:
            query_embedding: Query vector to search for.
            n_results: Optional limit on number of results to return.

        Returns:
            Cached results list if match found, None otherwise.
        """
        import time as _time

        start_time = _time.perf_counter() * 1000  # ms

        if not query_embedding or not self._entries:
            elapsed = _time.perf_counter() * 1000 - start_time
            self._misses += 1
            self._miss_latencies.append(elapsed)
            return None

        self._evict_expired()
        q_vec = np.asarray(query_embedding, dtype="float32")

        # Try index-based lookup first, then fall back to linear scan
        best_match = self._try_index_lookup(q_vec)
        if best_match is None:
            best_match = self._linear_scan_lookup(q_vec)

        if best_match is None:
            elapsed = _time.perf_counter() * 1000 - start_time
            self._misses += 1
            self._miss_latencies.append(elapsed)
            return None

        # Update access tracking
        score, entry = best_match
        self._update_entry_access(entry)

        # Record hit metrics
        elapsed = _time.perf_counter() * 1000 - start_time
        self._hits += 1
        self._hit_latencies.append(elapsed)

        logger.debug(
            "SemanticCache hit: %s score=%.3f last_access=%d", entry.query, score, entry.last_access
        )

        # Return results, optionally sliced
        results = entry.results
        if n_results is not None and isinstance(n_results, int):
            return results[:n_results]
        return results

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

            vectors = np.stack(
                [np.asarray(e.query_embedding, dtype="float32") for e in self._entries]
            )
            # normalize vectors for cosine similarity if using inner product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            dim = vectors.shape[1]
            self._dimension = dim
            if self._index_type == "hnsw":
                try:
                    idx = faiss.IndexHNSWFlat(dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
                except TypeError:
                    # Older FAISS bindings may not accept metric as positional arg
                    idx = faiss.IndexHNSWFlat(dim, self._hnsw_m)
                idx.hnsw.efConstruction = max(40, self._hnsw_m * 2)
                idx.hnsw.efSearch = max(50, self._hnsw_m * 2)
            else:
                idx = faiss.IndexFlatIP(dim)
            idx.add(vectors)
            self._index = idx
        except Exception as exc:
            logger.warning("Failed to build FAISS index for semantic cache: %s", exc)
            self._index = None


class RetrievalCacheService:
    """
    Unified caching service for retrieval operations.

    Consolidates both simple query caching (for testing) and semantic caching
    (for production) into a single service interface.
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        semantic_cache_enabled: bool = False,
        semantic_cache_ttl: int = 600,
        semantic_cache_threshold: float = 0.93,
        semantic_cache_max_entries: int = 512,
    ):
        """
        Initialize the caching service.

        Args:
            cache_dir: Directory for cache files
            semantic_cache_enabled: Whether to enable semantic caching
            semantic_cache_ttl: TTL for semantic cache entries
            semantic_cache_threshold: Similarity threshold for semantic cache
            semantic_cache_max_entries: Maximum entries in semantic cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Simple query cache (for testing and fast exact matches)
        self.query_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self.query_cache_file = self.cache_dir / "query_cache.json"

        # Semantic cache (for production similarity matching)
        self.semantic_cache: Optional[SemanticCache] = None
        if semantic_cache_enabled:
            semantic_cache_path = self.cache_dir / "semantic_cache.json"
            self.semantic_cache = SemanticCache(
                ttl_seconds=semantic_cache_ttl,
                similarity_threshold=semantic_cache_threshold,
                max_entries=semantic_cache_max_entries,
                cache_path=str(semantic_cache_path),
            )

        # Load existing query cache
        self._load_query_cache()

    def _load_query_cache(self) -> None:
        """Load simple query cache from disk."""
        if not self.query_cache_file.exists():
            return

        try:
            with self.query_cache_file.open("r", encoding="utf-8") as f:
                loaded_cache = json.load(f)

            # Convert string keys back to tuples
            self.query_cache = {}
            for key_str, value in loaded_cache.items():
                if key_str.startswith("(") and key_str.endswith(")"):
                    parts = key_str[1:-1].split(", ")
                    if len(parts) == 2:
                        key = (parts[0].strip("'\""), int(parts[1]))
                        self.query_cache[key] = value
        except Exception as exc:
            logger.warning("Failed to load query cache: %s", exc)
            self.query_cache = {}

    def save_query_cache(self) -> None:
        """Save simple query cache to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            serializable_cache = {str(k): v for k, v in self.query_cache.items()}
            with self.query_cache_file.open("w", encoding="utf-8") as f:
                json.dump(serializable_cache, f)
        except Exception as exc:
            logger.warning("Failed to save query cache: %s", exc)

    def lookup_query(
        self,
        query: str,
        top_k: int,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Look up cached results for a query.

        First checks semantic cache (if enabled), then falls back to exact match cache.

        Args:
            query: The query string
            top_k: Number of results requested
            query_embedding: Optional query embedding for semantic lookup

        Returns:
            Cached results if found, None otherwise
        """
        # Try semantic cache first
        if self.semantic_cache and query_embedding:
            cached = self.semantic_cache.lookup(query_embedding, n_results=top_k)
            if cached:
                logger.info("Semantic cache hit for query")
                return cached

        # Fall back to exact match cache
        cache_key = (query, top_k)
        if cache_key in self.query_cache:
            logger.info("Query cache hit for exact query match")
            return self.query_cache[cache_key]

        return None

    def cache_results(
        self,
        query: str,
        top_k: int,
        results: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Cache retrieval results.

        Args:
            query: The query string
            top_k: Number of results
            results: Results to cache
            query_embedding: Optional query embedding for semantic caching
        """
        if not results:
            return

        # Add to semantic cache if enabled
        if self.semantic_cache and query_embedding:
            self.semantic_cache.add(query, query_embedding, results)

        # Also add to simple query cache
        cache_key = (query, top_k)
        self.query_cache[cache_key] = results

    def clear(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        if self.semantic_cache:
            self.semantic_cache.clear()

        # Remove cache files
        try:
            if self.query_cache_file.exists():
                os.remove(self.query_cache_file)
        except OSError:
            pass

    def invalidate_for_documents(self, filenames: List[str]) -> None:
        """
        Invalidate cache entries related to specific documents.

        Note: This is a simple implementation that clears the entire cache.
        A more sophisticated version could track which queries relate to which documents.

        Args:
            filenames: List of document filenames that were modified
        """
        # For now, clear entire cache when documents change
        # Future: implement more granular invalidation
        if filenames:
            logger.info("Invalidating cache for %d documents", len(filenames))
            self.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get combined cache metrics for benchmarking.

        Returns:
            Dictionary with cache statistics including hit rate, latencies,
            and cache state information.
        """
        metrics: Dict[str, Any] = {
            "query_cache_entries": len(self.query_cache),
            "semantic_cache_enabled": self.semantic_cache is not None,
        }

        if self.semantic_cache:
            semantic_metrics = self.semantic_cache.get_metrics()
            metrics.update({"semantic_" + k: v for k, v in semantic_metrics.items()})
        else:
            # Provide defaults for non-semantic cache
            metrics.update(
                {
                    "semantic_total_queries": 0,
                    "semantic_hits": 0,
                    "semantic_misses": 0,
                    "semantic_hit_rate": 0.0,
                    "semantic_hit_rate_percent": 0.0,
                }
            )

        return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics for a fresh benchmark run."""
        if self.semantic_cache:
            self.semantic_cache.reset_metrics()


# Type alias for cache key tuples
from typing import Tuple
