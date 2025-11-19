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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_embedding": self.query_embedding,
            "results": self.results,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(
            query=data.get("query", ""),
            query_embedding=data.get("query_embedding", []),
            results=data.get("results", []),
            timestamp=data.get("timestamp", time.time()),
        )


class SemanticCache:
    """Semantic cache with TTL, similarity threshold, and optional persistence."""

    def __init__(
        self,
        ttl_seconds: int = 300,
        similarity_threshold: float = 0.92,
        max_entries: int = 256,
        cache_path: Optional[str] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cache_path = Path(cache_path) if cache_path else None
        self._entries: List[CacheEntry] = []
        if self.cache_path:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with self.cache_path.open('r', encoding='utf-8') as fh:
                raw = json.load(fh)
            self._entries = [CacheEntry.from_dict(item) for item in raw]
            logger.info("Loaded %d semantic cache entries", len(self._entries))
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
        # Remove oldest entries beyond capacity
        self._entries.sort(key=lambda entry: entry.timestamp, reverse=True)
        kept = self._entries[: self.max_entries]
        self._entries = kept

    def lookup(self, query_embedding: List[float]) -> Optional[List[Dict[str, Any]]]:
        if not query_embedding or not self._entries:
            return None
        self._evict_expired()
        q_vec = np.asarray(query_embedding, dtype='float32')
        best_match: Tuple[float, CacheEntry] | None = None
        for entry in self._entries:
            entry_vec = np.asarray(entry.query_embedding, dtype='float32')
            score = _cosine_similarity(q_vec, entry_vec)
            if score >= self.similarity_threshold:
                if not best_match or score > best_match[0]:
                    best_match = (score, entry)
        if best_match:
            logger.debug("Semantic cache hit with similarity %.3f", best_match[0])
            return best_match[1].results
        return None

    def add(self, query: str, query_embedding: List[float], results: List[Dict[str, Any]]) -> None:
        if not query_embedding or not results:
            return
        entry = CacheEntry(query=query, query_embedding=query_embedding, results=results)
        self._entries.append(entry)
        self._evict_expired()
        self._evict_lru()
        if self.cache_path:
            self._flush_to_disk()

    def clear(self) -> None:
        self._entries.clear()
        if self.cache_path and self.cache_path.exists():
            try:
                os.remove(self.cache_path)
            except OSError:
                pass
