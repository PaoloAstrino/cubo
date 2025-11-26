"""
Tiered Index Manager - Unified management of FAISS and BM25 indices.

This module provides a centralized manager for all retrieval indices with:
- Coordinated lifecycle (build, load, save, rebuild)
- Hot/cold index management for FAISS
- BM25 statistics tracking
- Health checks and diagnostics
- Versioned snapshots for rollback

The TieredIndexManager acts as a facade over FAISS and BM25 stores,
providing a unified interface for index operations.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.cubo.config import config
from src.cubo.utils.logger import logger


class IndexType(str, Enum):
    """Types of indices managed by the TieredIndexManager."""
    FAISS_HOT = "faiss_hot"
    FAISS_COLD = "faiss_cold"
    BM25 = "bm25"
    SCAFFOLD = "scaffold"
    SUMMARY = "summary"


class IndexState(str, Enum):
    """State of an index."""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    BUILDING = "building"
    ERROR = "error"
    STALE = "stale"


@dataclass
class IndexStats:
    """Statistics for an index."""
    index_type: IndexType
    state: IndexState
    document_count: int = 0
    vector_count: int = 0
    last_updated: Optional[datetime] = None
    size_bytes: int = 0
    build_time_ms: float = 0.0
    query_count: int = 0
    avg_query_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index_type": self.index_type.value,
            "state": self.state.value,
            "document_count": self.document_count,
            "vector_count": self.vector_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "size_bytes": self.size_bytes,
            "build_time_ms": self.build_time_ms,
            "query_count": self.query_count,
            "avg_query_time_ms": self.avg_query_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class IndexSnapshot:
    """A versioned snapshot of indices for rollback."""
    snapshot_id: str
    created_at: datetime
    indices: List[IndexType]
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
            "indices": [idx.value for idx in self.indices],
            "path": str(self.path),
            "metadata": self.metadata,
        }


class IndexHealthChecker:
    """Performs health checks on indices."""
    
    def __init__(self, manager: 'TieredIndexManager'):
        self._manager = manager
    
    def check_faiss_health(self) -> Dict[str, Any]:
        """Check FAISS index health."""
        result = {
            "healthy": False,
            "issues": [],
            "details": {},
        }
        
        stats = self._manager.get_stats(IndexType.FAISS_HOT)
        if stats.state != IndexState.READY:
            result["issues"].append(f"Hot index state: {stats.state.value}")
        
        cold_stats = self._manager.get_stats(IndexType.FAISS_COLD)
        if cold_stats.state not in (IndexState.READY, IndexState.UNINITIALIZED):
            result["issues"].append(f"Cold index state: {cold_stats.state.value}")
        
        result["details"]["hot_vectors"] = stats.vector_count
        result["details"]["cold_vectors"] = cold_stats.vector_count
        result["details"]["total_vectors"] = stats.vector_count + cold_stats.vector_count
        
        if not result["issues"]:
            result["healthy"] = True
        
        return result
    
    def check_bm25_health(self) -> Dict[str, Any]:
        """Check BM25 index health."""
        result = {
            "healthy": False,
            "issues": [],
            "details": {},
        }
        
        stats = self._manager.get_stats(IndexType.BM25)
        if stats.state != IndexState.READY:
            result["issues"].append(f"BM25 state: {stats.state.value}")
        
        result["details"]["document_count"] = stats.document_count
        
        if not result["issues"]:
            result["healthy"] = True
        
        return result
    
    def check_all(self) -> Dict[str, Any]:
        """Check health of all indices."""
        faiss_health = self.check_faiss_health()
        bm25_health = self.check_bm25_health()
        
        return {
            "overall_healthy": faiss_health["healthy"] and bm25_health["healthy"],
            "faiss": faiss_health,
            "bm25": bm25_health,
            "timestamp": datetime.utcnow().isoformat(),
        }


class TieredIndexManager:
    """
    Unified manager for FAISS and BM25 retrieval indices.
    
    Provides:
    - Coordinated index lifecycle management
    - Hot/cold tier management for FAISS
    - BM25 statistics tracking
    - Index health monitoring
    - Snapshot/rollback capabilities
    
    Usage:
        manager = TieredIndexManager(dimension=384)
        
        # Build indices
        manager.build_indices(vectors, texts, ids)
        
        # Search
        results = manager.search_faiss(query_vector, top_k=10)
        bm25_results = manager.search_bm25(query_text, top_k=10)
        
        # Health check
        health = manager.health_checker.check_all()
    """
    
    _instance: Optional['TieredIndexManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_dir: Optional[Path] = None,
        bm25_stats_path: Optional[str] = None,
        hot_fraction: float = 0.2,
        enable_snapshots: bool = True,
        max_snapshots: int = 3,
    ):
        """
        Initialize the tiered index manager.
        
        Args:
            dimension: Vector dimension for FAISS
            index_dir: Directory for index storage
            bm25_stats_path: Path for BM25 statistics
            hot_fraction: Fraction of vectors to keep in hot index
            enable_snapshots: Whether to enable snapshot functionality
            max_snapshots: Maximum number of snapshots to retain
        """
        if self._initialized:
            return
        
        self._dimension = dimension or config.get('embedding.dimension', 384)
        self._index_dir = Path(index_dir or config.get('vector_store_path', './faiss_store'))
        self._bm25_stats_path = bm25_stats_path or config.get('bm25_stats_path', 'data/bm25_stats.json')
        self._hot_fraction = hot_fraction
        self._enable_snapshots = enable_snapshots
        self._max_snapshots = max_snapshots
        
        # Index state tracking
        self._stats: Dict[IndexType, IndexStats] = {
            IndexType.FAISS_HOT: IndexStats(IndexType.FAISS_HOT, IndexState.UNINITIALIZED),
            IndexType.FAISS_COLD: IndexStats(IndexType.FAISS_COLD, IndexState.UNINITIALIZED),
            IndexType.BM25: IndexStats(IndexType.BM25, IndexState.UNINITIALIZED),
            IndexType.SCAFFOLD: IndexStats(IndexType.SCAFFOLD, IndexState.UNINITIALIZED),
            IndexType.SUMMARY: IndexStats(IndexType.SUMMARY, IndexState.UNINITIALIZED),
        }
        
        # Lazy-loaded index references
        self._faiss_manager = None
        self._bm25_store = None
        
        # Thread safety
        self._indices_lock = threading.RLock()
        
        # Snapshots
        self._snapshots: List[IndexSnapshot] = []
        self._snapshot_dir = self._index_dir / "snapshots"
        
        # Health checker
        self.health_checker = IndexHealthChecker(self)
        
        # Query statistics
        self._query_times: Dict[IndexType, List[float]] = {t: [] for t in IndexType}
        
        self._initialized = True
        logger.info(f"TieredIndexManager initialized (dimension={self._dimension}, index_dir={self._index_dir})")
    
    @property
    def faiss_manager(self):
        """Lazy-load FAISS index manager."""
        if self._faiss_manager is None:
            with self._indices_lock:
                if self._faiss_manager is None:
                    from src.cubo.indexing.faiss_index import FAISSIndexManager
                    self._faiss_manager = FAISSIndexManager(
                        dimension=self._dimension,
                        index_dir=self._index_dir,
                        hot_fraction=self._hot_fraction,
                    )
        return self._faiss_manager
    
    @property
    def bm25_store(self):
        """Lazy-load BM25 store."""
        if self._bm25_store is None:
            with self._indices_lock:
                if self._bm25_store is None:
                    from src.cubo.retrieval.bm25_store_factory import get_bm25_store
                    self._bm25_store = get_bm25_store()
                    
                    # Load existing stats if available
                    if os.path.exists(self._bm25_stats_path):
                        try:
                            self._bm25_store.load_stats(self._bm25_stats_path)
                            self._stats[IndexType.BM25].state = IndexState.READY
                            logger.info(f"Loaded BM25 stats from {self._bm25_stats_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load BM25 stats: {e}")
        return self._bm25_store
    
    def get_stats(self, index_type: IndexType) -> IndexStats:
        """Get statistics for an index."""
        return self._stats.get(index_type, IndexStats(index_type, IndexState.UNINITIALIZED))
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all indices."""
        return {
            idx_type.value: stats.to_dict() 
            for idx_type, stats in self._stats.items()
        }
    
    def build_faiss_index(
        self,
        vectors: List[List[float]],
        ids: List[str],
        append: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Build or update the FAISS index.
        
        Args:
            vectors: List of embedding vectors
            ids: List of document IDs
            append: Whether to append to existing index
            progress_callback: Optional callback(current, total)
        """
        with self._indices_lock:
            self._stats[IndexType.FAISS_HOT].state = IndexState.BUILDING
            self._stats[IndexType.FAISS_COLD].state = IndexState.BUILDING
            
            start_time = time.time()
            
            try:
                if progress_callback:
                    progress_callback(0, len(vectors))
                
                self.faiss_manager.build_indexes(vectors, ids, append=append)
                
                if progress_callback:
                    progress_callback(len(vectors), len(vectors))
                
                build_time = (time.time() - start_time) * 1000
                
                # Update stats
                hot_count = len(self.faiss_manager.hot_ids) if self.faiss_manager.hot_ids else 0
                cold_count = len(self.faiss_manager.cold_ids) if self.faiss_manager.cold_ids else 0
                
                self._stats[IndexType.FAISS_HOT].state = IndexState.READY
                self._stats[IndexType.FAISS_HOT].vector_count = hot_count
                self._stats[IndexType.FAISS_HOT].build_time_ms = build_time
                self._stats[IndexType.FAISS_HOT].last_updated = datetime.utcnow()
                
                self._stats[IndexType.FAISS_COLD].state = IndexState.READY if cold_count > 0 else IndexState.UNINITIALIZED
                self._stats[IndexType.FAISS_COLD].vector_count = cold_count
                self._stats[IndexType.FAISS_COLD].build_time_ms = build_time
                self._stats[IndexType.FAISS_COLD].last_updated = datetime.utcnow()
                
                logger.info(f"Built FAISS index: {hot_count} hot, {cold_count} cold vectors in {build_time:.1f}ms")
                
            except Exception as e:
                self._stats[IndexType.FAISS_HOT].state = IndexState.ERROR
                self._stats[IndexType.FAISS_COLD].state = IndexState.ERROR
                logger.error(f"Failed to build FAISS index: {e}")
                raise
    
    def build_bm25_index(
        self,
        documents: List[Dict],
        append: bool = False,
    ) -> None:
        """
        Build or update the BM25 index.
        
        Args:
            documents: List of document dicts with 'doc_id' and 'text'
            append: Whether to append to existing index
        """
        with self._indices_lock:
            self._stats[IndexType.BM25].state = IndexState.BUILDING
            start_time = time.time()
            
            try:
                if append:
                    self.bm25_store.add_documents(documents)
                else:
                    self.bm25_store.index_documents(documents)
                
                build_time = (time.time() - start_time) * 1000
                
                self._stats[IndexType.BM25].state = IndexState.READY
                self._stats[IndexType.BM25].document_count = len(self.bm25_store.docs)
                self._stats[IndexType.BM25].build_time_ms = build_time
                self._stats[IndexType.BM25].last_updated = datetime.utcnow()
                
                logger.info(f"Built BM25 index: {len(documents)} documents in {build_time:.1f}ms")
                
            except Exception as e:
                self._stats[IndexType.BM25].state = IndexState.ERROR
                logger.error(f"Failed to build BM25 index: {e}")
                raise
    
    def build_all(
        self,
        vectors: List[List[float]],
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Build all indices from documents.
        
        Args:
            vectors: List of embedding vectors
            texts: List of document texts
            ids: List of document IDs
            metadatas: Optional list of metadata dicts
            progress_callback: Optional callback(phase, current, total)
        """
        total_docs = len(ids)
        
        # Build FAISS index
        if progress_callback:
            progress_callback("faiss", 0, total_docs)
        self.build_faiss_index(vectors, ids)
        if progress_callback:
            progress_callback("faiss", total_docs, total_docs)
        
        # Build BM25 index
        documents = [{"doc_id": doc_id, "text": text} for doc_id, text in zip(ids, texts)]
        if progress_callback:
            progress_callback("bm25", 0, total_docs)
        self.build_bm25_index(documents)
        if progress_callback:
            progress_callback("bm25", total_docs, total_docs)
    
    def search_faiss(
        self,
        query_vector: List[float],
        top_k: int = 10,
        search_hot_only: bool = False,
    ) -> List[Dict]:
        """
        Search FAISS index for nearest neighbors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            search_hot_only: Only search hot index (faster but less complete)
            
        Returns:
            List of results with 'id' and 'distance'
        """
        start_time = time.time()
        
        try:
            results = self.faiss_manager.search(query_vector, k=top_k)
            
            query_time = (time.time() - start_time) * 1000
            self._record_query_time(IndexType.FAISS_HOT, query_time)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        docs: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Search BM25 index.
        
        Args:
            query: Query text
            top_k: Number of results to return
            docs: Optional subset of documents to search
            
        Returns:
            List of results with 'doc_id', 'similarity', 'text', 'metadata'
        """
        start_time = time.time()
        
        try:
            results = self.bm25_store.search(query, top_k=top_k, docs=docs)
            
            query_time = (time.time() - start_time) * 1000
            self._record_query_time(IndexType.BM25, query_time)
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _record_query_time(self, index_type: IndexType, time_ms: float) -> None:
        """Record query time for statistics."""
        times = self._query_times[index_type]
        times.append(time_ms)
        
        # Keep last 100 query times
        if len(times) > 100:
            times.pop(0)
        
        # Update stats
        self._stats[index_type].query_count += 1
        self._stats[index_type].avg_query_time_ms = sum(times) / len(times)
    
    def save_indices(self, path: Optional[Path] = None) -> None:
        """
        Save all indices to disk.
        
        Args:
            path: Optional custom path (uses default if None)
        """
        save_path = Path(path) if path else self._index_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        with self._indices_lock:
            # Save FAISS indices
            if self._faiss_manager:
                self.faiss_manager.save(save_path)
                logger.info(f"Saved FAISS indices to {save_path}")
            
            # Save BM25 stats
            if self._bm25_store:
                bm25_path = save_path / "bm25_stats.json"
                self.bm25_store.save_stats(str(bm25_path))
                logger.info(f"Saved BM25 stats to {bm25_path}")
            
            # Save manager metadata
            self._save_metadata(save_path)
    
    def load_indices(self, path: Optional[Path] = None) -> bool:
        """
        Load indices from disk.
        
        Args:
            path: Optional custom path (uses default if None)
            
        Returns:
            True if loaded successfully
        """
        load_path = Path(path) if path else self._index_dir
        
        if not load_path.exists():
            logger.warning(f"Index directory not found: {load_path}")
            return False
        
        with self._indices_lock:
            try:
                # Load FAISS indices
                self._stats[IndexType.FAISS_HOT].state = IndexState.LOADING
                self._stats[IndexType.FAISS_COLD].state = IndexState.LOADING
                
                self.faiss_manager.load(load_path)
                
                hot_count = len(self.faiss_manager.hot_ids) if self.faiss_manager.hot_ids else 0
                cold_count = len(self.faiss_manager.cold_ids) if self.faiss_manager.cold_ids else 0
                
                self._stats[IndexType.FAISS_HOT].state = IndexState.READY
                self._stats[IndexType.FAISS_HOT].vector_count = hot_count
                self._stats[IndexType.FAISS_COLD].state = IndexState.READY if cold_count > 0 else IndexState.UNINITIALIZED
                self._stats[IndexType.FAISS_COLD].vector_count = cold_count
                
                logger.info(f"Loaded FAISS indices: {hot_count} hot, {cold_count} cold")
                
                # Load BM25 stats
                bm25_path = load_path / "bm25_stats.json"
                if bm25_path.exists():
                    self._stats[IndexType.BM25].state = IndexState.LOADING
                    self.bm25_store.load_stats(str(bm25_path))
                    self._stats[IndexType.BM25].state = IndexState.READY
                    self._stats[IndexType.BM25].document_count = len(self.bm25_store.docs)
                    logger.info(f"Loaded BM25 stats from {bm25_path}")
                
                return True
                
            except Exception as e:
                self._stats[IndexType.FAISS_HOT].state = IndexState.ERROR
                self._stats[IndexType.FAISS_COLD].state = IndexState.ERROR
                self._stats[IndexType.BM25].state = IndexState.ERROR
                logger.error(f"Failed to load indices: {e}")
                return False
    
    def _save_metadata(self, path: Path) -> None:
        """Save manager metadata."""
        metadata = {
            "dimension": self._dimension,
            "hot_fraction": self._hot_fraction,
            "stats": self.get_all_stats(),
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        meta_path = path / "index_manager_metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def create_snapshot(self, name: Optional[str] = None) -> Optional[IndexSnapshot]:
        """
        Create a snapshot of current indices.
        
        Args:
            name: Optional snapshot name
            
        Returns:
            IndexSnapshot if successful, None otherwise
        """
        if not self._enable_snapshots:
            logger.warning("Snapshots are disabled")
            return None
        
        snapshot_id = name or f"snapshot_{int(time.time())}"
        snapshot_path = self._snapshot_dir / snapshot_id
        
        try:
            snapshot_path.mkdir(parents=True, exist_ok=True)
            self.save_indices(snapshot_path)
            
            snapshot = IndexSnapshot(
                snapshot_id=snapshot_id,
                created_at=datetime.utcnow(),
                indices=[IndexType.FAISS_HOT, IndexType.FAISS_COLD, IndexType.BM25],
                path=snapshot_path,
                metadata=self.get_all_stats(),
            )
            
            self._snapshots.append(snapshot)
            self._cleanup_old_snapshots()
            
            logger.info(f"Created snapshot: {snapshot_id}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore indices from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            
        Returns:
            True if restored successfully
        """
        snapshot = next((s for s in self._snapshots if s.snapshot_id == snapshot_id), None)
        
        if not snapshot:
            logger.error(f"Snapshot not found: {snapshot_id}")
            return False
        
        return self.load_indices(snapshot.path)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        return [s.to_dict() for s in self._snapshots]
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond max_snapshots."""
        while len(self._snapshots) > self._max_snapshots:
            oldest = self._snapshots.pop(0)
            try:
                shutil.rmtree(oldest.path)
                logger.info(f"Removed old snapshot: {oldest.snapshot_id}")
            except Exception as e:
                logger.warning(f"Failed to remove snapshot {oldest.snapshot_id}: {e}")
    
    def reset(self) -> None:
        """Reset all indices to uninitialized state."""
        with self._indices_lock:
            self._faiss_manager = None
            self._bm25_store = None
            
            for idx_type in IndexType:
                self._stats[idx_type] = IndexStats(idx_type, IndexState.UNINITIALIZED)
            
            logger.info("Reset all indices")
    
    def get_total_document_count(self) -> int:
        """Get total number of indexed documents."""
        faiss_count = (
            self._stats[IndexType.FAISS_HOT].vector_count +
            self._stats[IndexType.FAISS_COLD].vector_count
        )
        return max(faiss_count, self._stats[IndexType.BM25].document_count)


# Global instance accessor
_tiered_index_manager: Optional[TieredIndexManager] = None


def get_tiered_index_manager(dimension: Optional[int] = None) -> TieredIndexManager:
    """Get the global tiered index manager instance."""
    global _tiered_index_manager
    if _tiered_index_manager is None:
        _tiered_index_manager = TieredIndexManager(dimension=dimension)
    return _tiered_index_manager
