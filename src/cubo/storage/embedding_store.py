"""
On-disk embedding persistence for vector stores.

Provides memory-efficient storage of embeddings using numpy's memory-mapped
arrays (mmap) and optional sharding for large collections.

Supported storage modes:
- 'memory': In-memory storage (default, fast but uses RAM)
- 'npy': Single numpy file per embedding (simple, good for small collections)
- 'npy_sharded': Sharded numpy files (good for laptop mode, reduces memory)
- 'mmap': Memory-mapped single file (best for large collections)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


class EmbeddingCache:
    """LRU cache for embeddings to avoid repeated disk reads."""
    
    def __init__(self, max_size: int = 512):
        """Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, doc_id: str) -> Optional[np.ndarray]:
        """Get an embedding from cache, updating LRU order."""
        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
                self._hits += 1
                return self._cache[doc_id]
            self._misses += 1
            return None
    
    def put(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add an embedding to cache, evicting oldest if full."""
        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
                self._cache[doc_id] = embedding
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[doc_id] = embedding
    
    def remove(self, doc_id: str) -> None:
        """Remove an embedding from cache."""
        with self._lock:
            self._cache.pop(doc_id, None)
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_batch(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get multiple embeddings, returning those found in cache."""
        result = {}
        with self._lock:
            for doc_id in doc_ids:
                if doc_id in self._cache:
                    self._cache.move_to_end(doc_id)
                    result[doc_id] = self._cache[doc_id]
                    self._hits += 1
                else:
                    self._misses += 1
        return result
    
    def put_batch(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Add multiple embeddings to cache."""
        with self._lock:
            for doc_id, embedding in embeddings.items():
                if doc_id in self._cache:
                    self._cache.move_to_end(doc_id)
                else:
                    if len(self._cache) >= self._max_size:
                        self._cache.popitem(last=False)
                self._cache[doc_id] = embedding
    
    @property
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'size': len(self._cache),
                'max_size': self._max_size,
                'hit_rate_pct': round(hit_rate, 2)
            }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __bool__(self) -> bool:
        return True  # Cache is always truthy


class EmbeddingStore:
    """Base interface for embedding storage backends."""
    
    def add(self, doc_id: str, embedding: Union[List[float], np.ndarray]) -> None:
        """Store an embedding."""
        raise NotImplementedError()
    
    def add_batch(self, embeddings: Dict[str, Union[List[float], np.ndarray]]) -> None:
        """Store multiple embeddings."""
        for doc_id, emb in embeddings.items():
            self.add(doc_id, emb)
    
    def get(self, doc_id: str) -> Optional[np.ndarray]:
        """Retrieve an embedding."""
        raise NotImplementedError()
    
    def get_batch(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve multiple embeddings."""
        result = {}
        for doc_id in doc_ids:
            emb = self.get(doc_id)
            if emb is not None:
                result[doc_id] = emb
        return result
    
    def delete(self, doc_id: str) -> None:
        """Delete an embedding."""
        raise NotImplementedError()
    
    def delete_batch(self, doc_ids: List[str]) -> None:
        """Delete multiple embeddings."""
        for doc_id in doc_ids:
            self.delete(doc_id)
    
    def clear(self) -> None:
        """Clear all embeddings."""
        raise NotImplementedError()
    
    def count(self) -> int:
        """Return number of stored embeddings."""
        raise NotImplementedError()
    
    def keys(self) -> Iterator[str]:
        """Iterate over all document IDs."""
        raise NotImplementedError()
    
    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over all (doc_id, embedding) pairs."""
        raise NotImplementedError()
    
    def close(self) -> None:
        """Release any resources."""
        pass


class InMemoryEmbeddingStore(EmbeddingStore):
    """Simple in-memory embedding storage (default behavior)."""
    
    def __init__(self, dtype: str = 'float32'):
        self._embeddings: Dict[str, np.ndarray] = {}
        self._dtype = np.dtype(dtype)
        self._lock = threading.Lock()
    
    def add(self, doc_id: str, embedding: Union[List[float], np.ndarray]) -> None:
        arr = np.asarray(embedding, dtype=self._dtype)
        with self._lock:
            self._embeddings[doc_id] = arr
    
    def add_batch(self, embeddings: Dict[str, Union[List[float], np.ndarray]]) -> None:
        with self._lock:
            for doc_id, emb in embeddings.items():
                self._embeddings[doc_id] = np.asarray(emb, dtype=self._dtype)
    
    def get(self, doc_id: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._embeddings.get(doc_id)
    
    def get_batch(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        with self._lock:
            return {did: self._embeddings[did] for did in doc_ids if did in self._embeddings}
    
    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._embeddings.pop(doc_id, None)
    
    def delete_batch(self, doc_ids: List[str]) -> None:
        with self._lock:
            for doc_id in doc_ids:
                self._embeddings.pop(doc_id, None)
    
    def clear(self) -> None:
        with self._lock:
            self._embeddings.clear()
    
    def count(self) -> int:
        return len(self._embeddings)
    
    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._embeddings.keys()))
    
    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        with self._lock:
            items_copy = list(self._embeddings.items())
        yield from items_copy


class ShardedEmbeddingStore(EmbeddingStore):
    """Sharded on-disk embedding storage using numpy files.
    
    Embeddings are stored in shards of configurable size. Each shard is a
    numpy file containing a matrix of embeddings plus a JSON index mapping
    doc_ids to row indices. This approach:
    - Reduces memory usage by loading only needed shards
    - Supports efficient batch operations
    - Works well with float16 for ~50% memory reduction
    """
    
    def __init__(
        self,
        storage_dir: Path,
        shard_size: int = 1000,
        dtype: str = 'float32',
        cache_size: int = 512,
        enable_cache: bool = True
    ):
        """Initialize sharded embedding store.
        
        Args:
            storage_dir: Directory to store shard files
            shard_size: Maximum embeddings per shard
            dtype: Numpy dtype ('float32' or 'float16')
            cache_size: LRU cache size for embeddings
            enable_cache: Whether to enable LRU caching
        """
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        
        self._shard_size = shard_size
        self._dtype = np.dtype(dtype)
        self._lock = threading.Lock()
        
        # Cache for recently accessed embeddings
        self._cache = EmbeddingCache(cache_size) if enable_cache else None
        
        # Index: doc_id -> (shard_id, row_index)
        self._index: Dict[str, Tuple[int, int]] = {}
        
        # Current write shard
        self._current_shard_id = 0
        self._current_shard_count = 0
        
        # Load existing index if present
        self._load_index()
    
    def _index_path(self) -> Path:
        return self._dir / 'embedding_index.json'
    
    def _shard_path(self, shard_id: int) -> Path:
        return self._dir / f'embeddings_shard_{shard_id:04d}.npy'
    
    def _load_index(self) -> None:
        """Load index from disk."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self._index = {k: tuple(v) for k, v in data.get('index', {}).items()}
                    self._current_shard_id = data.get('current_shard_id', 0)
                    self._current_shard_count = data.get('current_shard_count', 0)
            except Exception:
                self._index = {}
                self._current_shard_id = 0
                self._current_shard_count = 0
    
    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self._index_path()
        data = {
            'index': {k: list(v) for k, v in self._index.items()},
            'current_shard_id': self._current_shard_id,
            'current_shard_count': self._current_shard_count
        }
        # Atomic write
        tmp_path = index_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        tmp_path.replace(index_path)
    
    def _load_shard(self, shard_id: int) -> Optional[np.ndarray]:
        """Load a shard from disk."""
        path = self._shard_path(shard_id)
        if path.exists():
            return np.load(path, allow_pickle=False)
        return None
    
    def _save_shard(self, shard_id: int, data: np.ndarray) -> None:
        """Save a shard to disk."""
        path = self._shard_path(shard_id)
        # Use a unique temp file to avoid conflicts
        tmp_path = path.parent / f'.tmp_shard_{shard_id}_{os.getpid()}.npy'
        try:
            np.save(str(tmp_path), data)
            # On Windows, target must not exist for replace
            if path.exists():
                path.unlink()
            tmp_path.rename(path)
        except Exception:
            # Clean up temp file on error
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise
    
    def add(self, doc_id: str, embedding: Union[List[float], np.ndarray]) -> None:
        arr = np.asarray(embedding, dtype=self._dtype)
        
        with self._lock:
            # If already exists, update in place
            if doc_id in self._index:
                shard_id, row_idx = self._index[doc_id]
                shard = self._load_shard(shard_id)
                if shard is not None and row_idx < len(shard):
                    shard[row_idx] = arr
                    self._save_shard(shard_id, shard)
                    if self._cache:
                        self._cache.put(doc_id, arr)
                    return
            
            # Check if current shard is full
            if self._current_shard_count >= self._shard_size:
                self._current_shard_id += 1
                self._current_shard_count = 0
            
            # Load or create current shard
            shard = self._load_shard(self._current_shard_id)
            if shard is None:
                # Create new shard with initial embedding
                shard = arr.reshape(1, -1)
            else:
                # Append to existing shard
                shard = np.vstack([shard, arr.reshape(1, -1)])
            
            # Update index
            row_idx = len(shard) - 1
            self._index[doc_id] = (self._current_shard_id, row_idx)
            self._current_shard_count = len(shard)
            
            # Save shard and index
            self._save_shard(self._current_shard_id, shard)
            self._save_index()
            
            # Update cache
            if self._cache:
                self._cache.put(doc_id, arr)
    
    def add_batch(self, embeddings: Dict[str, Union[List[float], np.ndarray]]) -> None:
        """Efficiently add multiple embeddings."""
        if not embeddings:
            return
        
        with self._lock:
            # Separate updates from new additions
            updates: Dict[int, List[Tuple[str, int, np.ndarray]]] = {}
            new_embeddings: List[Tuple[str, np.ndarray]] = []
            
            for doc_id, emb in embeddings.items():
                arr = np.asarray(emb, dtype=self._dtype)
                if doc_id in self._index:
                    shard_id, row_idx = self._index[doc_id]
                    if shard_id not in updates:
                        updates[shard_id] = []
                    updates[shard_id].append((doc_id, row_idx, arr))
                else:
                    new_embeddings.append((doc_id, arr))
            
            # Process updates by shard
            for shard_id, update_list in updates.items():
                shard = self._load_shard(shard_id)
                if shard is not None:
                    for doc_id, row_idx, arr in update_list:
                        if row_idx < len(shard):
                            shard[row_idx] = arr
                            if self._cache:
                                self._cache.put(doc_id, arr)
                    self._save_shard(shard_id, shard)
            
            # Process new additions
            for doc_id, arr in new_embeddings:
                # Check if current shard is full
                if self._current_shard_count >= self._shard_size:
                    self._current_shard_id += 1
                    self._current_shard_count = 0
                
                # Load or create current shard
                shard = self._load_shard(self._current_shard_id)
                if shard is None:
                    shard = arr.reshape(1, -1)
                else:
                    shard = np.vstack([shard, arr.reshape(1, -1)])
                
                # Update index
                row_idx = len(shard) - 1
                self._index[doc_id] = (self._current_shard_id, row_idx)
                self._current_shard_count = len(shard)
                
                self._save_shard(self._current_shard_id, shard)
                
                if self._cache:
                    self._cache.put(doc_id, arr)
            
            # Save index once at the end
            self._save_index()
    
    def get(self, doc_id: str) -> Optional[np.ndarray]:
        # Check cache first
        if self._cache:
            cached = self._cache.get(doc_id)
            if cached is not None:
                return cached
        
        with self._lock:
            if doc_id not in self._index:
                return None
            
            shard_id, row_idx = self._index[doc_id]
            shard = self._load_shard(shard_id)
            
            if shard is None or row_idx >= len(shard):
                return None
            
            embedding = shard[row_idx].copy()
            
            # Update cache
            if self._cache:
                self._cache.put(doc_id, embedding)
            
            return embedding
    
    def get_batch(self, doc_ids: List[str]) -> Dict[str, np.ndarray]:
        result = {}
        missing = []
        
        # Check cache first
        if self._cache:
            cached = self._cache.get_batch(doc_ids)
            result.update(cached)
            missing = [did for did in doc_ids if did not in result]
        else:
            missing = doc_ids
        
        if not missing:
            return result
        
        with self._lock:
            # Group by shard for efficient loading
            shard_requests: Dict[int, List[Tuple[str, int]]] = {}
            for doc_id in missing:
                if doc_id in self._index:
                    shard_id, row_idx = self._index[doc_id]
                    if shard_id not in shard_requests:
                        shard_requests[shard_id] = []
                    shard_requests[shard_id].append((doc_id, row_idx))
            
            # Load each needed shard once
            cache_updates = {}
            for shard_id, requests in shard_requests.items():
                shard = self._load_shard(shard_id)
                if shard is not None:
                    for doc_id, row_idx in requests:
                        if row_idx < len(shard):
                            emb = shard[row_idx].copy()
                            result[doc_id] = emb
                            cache_updates[doc_id] = emb
            
            # Update cache in batch
            if self._cache and cache_updates:
                self._cache.put_batch(cache_updates)
        
        return result
    
    def delete(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._index:
                # Note: We don't actually remove from shard (would require rebuild)
                # Just remove from index. Space will be reclaimed on compaction.
                del self._index[doc_id]
                self._save_index()
                
            if self._cache:
                self._cache.remove(doc_id)
    
    def delete_batch(self, doc_ids: List[str]) -> None:
        with self._lock:
            changed = False
            for doc_id in doc_ids:
                if doc_id in self._index:
                    del self._index[doc_id]
                    changed = True
                if self._cache:
                    self._cache.remove(doc_id)
            if changed:
                self._save_index()
    
    def clear(self) -> None:
        with self._lock:
            # Remove all shard files
            for shard_file in self._dir.glob('embeddings_shard_*.npy'):
                try:
                    shard_file.unlink()
                except Exception:
                    pass
            
            # Clear index
            self._index.clear()
            self._current_shard_id = 0
            self._current_shard_count = 0
            self._save_index()
            
            if self._cache:
                self._cache.clear()
    
    def count(self) -> int:
        return len(self._index)
    
    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._index.keys()))
    
    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate all embeddings, loading shards as needed."""
        with self._lock:
            keys = list(self._index.keys())
        
        # Batch by shard for efficiency
        for doc_id in keys:
            emb = self.get(doc_id)
            if emb is not None:
                yield doc_id, emb
    
    def compact(self) -> None:
        """Compact storage by removing deleted entries.
        
        This rebuilds all shards, removing gaps from deleted entries.
        Should be called periodically if many deletions occur.
        """
        with self._lock:
            if not self._index:
                self.clear()
                return
            
            # Collect all valid embeddings
            all_embeddings: List[Tuple[str, np.ndarray]] = []
            for doc_id in list(self._index.keys()):
                shard_id, row_idx = self._index[doc_id]
                shard = self._load_shard(shard_id)
                if shard is not None and row_idx < len(shard):
                    all_embeddings.append((doc_id, shard[row_idx].copy()))
            
            # Clear and rebuild
            for shard_file in self._dir.glob('embeddings_shard_*.npy'):
                try:
                    shard_file.unlink()
                except Exception:
                    pass
            
            self._index.clear()
            self._current_shard_id = 0
            self._current_shard_count = 0
            
            if self._cache:
                self._cache.clear()
            
            # Re-add all embeddings
            for doc_id, emb in all_embeddings:
                if self._current_shard_count >= self._shard_size:
                    self._current_shard_id += 1
                    self._current_shard_count = 0
                
                shard = self._load_shard(self._current_shard_id)
                if shard is None:
                    shard = emb.reshape(1, -1)
                else:
                    shard = np.vstack([shard, emb.reshape(1, -1)])
                
                row_idx = len(shard) - 1
                self._index[doc_id] = (self._current_shard_id, row_idx)
                self._current_shard_count = len(shard)
                self._save_shard(self._current_shard_id, shard)
            
            self._save_index()
    
    @property
    def cache_stats(self) -> Optional[Dict[str, int]]:
        """Return cache statistics if caching is enabled."""
        return self._cache.stats if self._cache else None


class MmapEmbeddingStore(EmbeddingStore):
    """Memory-mapped embedding storage for large collections.
    
    Uses a single pre-allocated memory-mapped file for all embeddings.
    Best for:
    - Very large collections (millions of embeddings)
    - When embeddings size exceeds available RAM
    - Read-heavy workloads
    
    Note: Requires knowing dimension upfront and has fixed capacity.
    """
    
    def __init__(
        self,
        storage_path: Path,
        dimension: int,
        max_embeddings: int = 1_000_000,
        dtype: str = 'float32',
        cache_size: int = 512,
        enable_cache: bool = True
    ):
        """Initialize memory-mapped embedding store.
        
        Args:
            storage_path: Path to the mmap file
            dimension: Embedding dimension (required)
            max_embeddings: Maximum number of embeddings to store
            dtype: Numpy dtype ('float32' or 'float16')
            cache_size: LRU cache size for embeddings
            enable_cache: Whether to enable LRU caching
        """
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        self._dim = dimension
        self._max = max_embeddings
        self._dtype = np.dtype(dtype)
        self._lock = threading.Lock()
        
        # Cache
        self._cache = EmbeddingCache(cache_size) if enable_cache else None
        
        # Index: doc_id -> row_index
        self._index: Dict[str, int] = {}
        self._next_row = 0
        self._deleted_rows: List[int] = []  # Reusable slots
        
        # Create or load mmap
        self._mmap: Optional[np.memmap] = None
        self._initialize_storage()
    
    def _index_path(self) -> Path:
        return self._path.with_suffix('.index.json')
    
    def _initialize_storage(self) -> None:
        """Initialize or load existing storage."""
        # Load index if exists
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self._index = data.get('index', {})
                    self._next_row = data.get('next_row', 0)
                    self._deleted_rows = data.get('deleted_rows', [])
            except Exception:
                pass
        
        # Create or open mmap file
        shape = (self._max, self._dim)
        if self._path.exists():
            self._mmap = np.memmap(self._path, dtype=self._dtype, mode='r+', shape=shape)
        else:
            self._mmap = np.memmap(self._path, dtype=self._dtype, mode='w+', shape=shape)
    
    def _save_index(self) -> None:
        """Save index to disk."""
        data = {
            'index': self._index,
            'next_row': self._next_row,
            'deleted_rows': self._deleted_rows
        }
        tmp_path = self._index_path().with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        tmp_path.replace(self._index_path())
    
    def add(self, doc_id: str, embedding: Union[List[float], np.ndarray]) -> None:
        arr = np.asarray(embedding, dtype=self._dtype)
        
        with self._lock:
            if doc_id in self._index:
                # Update existing
                row_idx = self._index[doc_id]
            elif self._deleted_rows:
                # Reuse deleted slot
                row_idx = self._deleted_rows.pop()
            else:
                # Allocate new row
                if self._next_row >= self._max:
                    raise RuntimeError("Mmap storage full - increase max_embeddings")
                row_idx = self._next_row
                self._next_row += 1
            
            self._mmap[row_idx] = arr
            self._mmap.flush()
            self._index[doc_id] = row_idx
            self._save_index()
            
            if self._cache:
                self._cache.put(doc_id, arr)
    
    def get(self, doc_id: str) -> Optional[np.ndarray]:
        if self._cache:
            cached = self._cache.get(doc_id)
            if cached is not None:
                return cached
        
        with self._lock:
            if doc_id not in self._index:
                return None
            
            row_idx = self._index[doc_id]
            embedding = np.array(self._mmap[row_idx])
            
            if self._cache:
                self._cache.put(doc_id, embedding)
            
            return embedding
    
    def delete(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._index:
                row_idx = self._index.pop(doc_id)
                self._deleted_rows.append(row_idx)
                self._save_index()
                
            if self._cache:
                self._cache.remove(doc_id)
    
    def clear(self) -> None:
        with self._lock:
            self._index.clear()
            self._next_row = 0
            self._deleted_rows.clear()
            self._save_index()
            
            # Zero out mmap
            self._mmap[:] = 0
            self._mmap.flush()
            
            if self._cache:
                self._cache.clear()
    
    def count(self) -> int:
        return len(self._index)
    
    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._index.keys()))
    
    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        with self._lock:
            keys = list(self._index.keys())
        
        for doc_id in keys:
            emb = self.get(doc_id)
            if emb is not None:
                yield doc_id, emb
    
    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None


def create_embedding_store(
    mode: str = 'memory',
    storage_dir: Optional[Path] = None,
    dimension: int = 1536,
    dtype: str = 'float32',
    cache_size: int = 512,
    shard_size: int = 1000,
    max_embeddings: int = 1_000_000,
    enable_cache: bool = True
) -> EmbeddingStore:
    """Factory function to create an embedding store.
    
    Args:
        mode: Storage mode ('memory', 'npy_sharded', 'mmap')
        storage_dir: Directory for on-disk storage
        dimension: Embedding dimension (required for mmap)
        dtype: Numpy dtype ('float32' or 'float16')
        cache_size: LRU cache size
        shard_size: Embeddings per shard (for npy_sharded)
        max_embeddings: Max capacity (for mmap)
        enable_cache: Whether to enable LRU caching
    
    Returns:
        EmbeddingStore instance
    """
    if mode == 'memory':
        return InMemoryEmbeddingStore(dtype=dtype)
    
    if storage_dir is None:
        raise ValueError(f"storage_dir required for mode '{mode}'")
    
    storage_dir = Path(storage_dir)
    
    if mode in ('npy', 'npy_sharded'):
        return ShardedEmbeddingStore(
            storage_dir=storage_dir,
            shard_size=shard_size,
            dtype=dtype,
            cache_size=cache_size,
            enable_cache=enable_cache
        )
    
    if mode == 'mmap':
        return MmapEmbeddingStore(
            storage_path=storage_dir / 'embeddings.mmap',
            dimension=dimension,
            max_embeddings=max_embeddings,
            dtype=dtype,
            cache_size=cache_size,
            enable_cache=enable_cache
        )
    
    raise ValueError(f"Unknown embedding storage mode: {mode}")
