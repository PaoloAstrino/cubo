"""
Vector store abstraction with FAISS as the primary backend.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from src.cubo.config import config

# Module-level executor for async operations (lazy initialized)
_promotion_executor: Optional[ThreadPoolExecutor] = None
_promotion_lock = threading.Lock()


def _get_promotion_executor() -> ThreadPoolExecutor:
    """Get or create the shared ThreadPoolExecutor for async promotions."""
    global _promotion_executor
    if _promotion_executor is None:
        with _promotion_lock:
            if _promotion_executor is None:
                _promotion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="faiss_promote")
    return _promotion_executor


class VectorStore:
    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()

    def get(self, include=None, where=None, ids=None):
        raise NotImplementedError()

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        raise NotImplementedError()

    def save(self, path: Optional[Path] = None) -> None:
        pass

    def load(self, path: Optional[Path] = None) -> None:
        pass

    def reset(self) -> None:
        raise NotImplementedError()
    def delete(self, ids=None) -> None:
        """Delete ids from the store if supported. Default: NotImplemented."""
        raise NotImplementedError()


class FaissStore(VectorStore):
    """FAISS-backed vector store with async hot/cold index promotion.
    
    The hot/cold architecture keeps frequently accessed vectors in a fast
    in-memory index (hot) while less-accessed vectors remain in a compressed
    on-disk index (cold). Promotions happen asynchronously to avoid blocking
    query operations.
    """
    
    def __init__(self, dimension: int, index_dir: Optional[Path] = None, index_root: Optional[Path] = None):
        self.dimension = dimension
        self.index_dir = Path(index_dir) if index_dir else Path(config.get('vector_store_path', './faiss_store'))
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        self._index = FAISSIndexManager(dimension, index_dir=self.index_dir, index_root=index_root)
        self.index_root = index_root
        # local maps: id -> text/metadata
        self._docs: Dict[str, str] = {}
        self._metas: Dict[str, Dict] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._access_counts: Dict[str, int] = {}
        # Configure hot fraction from config
        from src.cubo.config import config as _config
        self.hot_fraction = float(_config.get('vector_index.hot_ratio', 0.2))
        self._index.hot_fraction = self.hot_fraction
        
        # Async promotion state
        self._pending_promotions: Set[str] = set()
        self._promotion_lock = threading.Lock()
        self._promotion_in_progress = False

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        # Persist to FAISS and store metadata locally
        if not embeddings or not ids:
            return
        self._index.build_indexes(embeddings, ids, append=True)
        # Store embeddings for potential rebuilds (e.g., promotion to hot)
        for i, did in enumerate(ids):
            self._embeddings[did] = embeddings[i]
            self._access_counts.setdefault(did, 0)
        for i, did in enumerate(ids):
            self._docs[did] = documents[i] if documents and i < len(documents) else ''
            self._metas[did] = metadatas[i] if metadatas and i < len(metadatas) else {}

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None, where=None, ids=None):
        # Return structure: 'ids', 'documents', 'metadatas' arrays
        ids_out = []
        docs = []
        metas = []
        if ids:
            for did in ids:
                if did in self._docs:
                    ids_out.append(did)
                    docs.append(self._docs.get(did, ''))
                    metas.append(self._metas.get(did, {}))
        else:
            # If a 'where' filter is provided, apply it against stored metadata
            if where and isinstance(where, dict):
                for did, doc in self._docs.items():
                    meta = self._metas.get(did, {})
                    match = True
                    for key, val in where.items():
                        # Support simple equality checks
                        if meta.get(key) != val:
                            match = False
                            break
                    if match:
                        ids_out.append(did)
                        docs.append(doc)
                        metas.append(meta)
            else:
                for did, doc in self._docs.items():
                    ids_out.append(did)
                    docs.append(doc)
                    metas.append(self._metas.get(did, {}))
        return {"ids": ids_out, "documents": [docs], "metadatas": [metas]}

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        """Query FAISS for nearest neighbors with async hot promotion."""
        if not query_embeddings or self.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        results = self._index.search(query_embeddings[0], k=n_results)
        # map results to documents/metas/ids/distance
        docs = []
        metas = []
        dists = []
        ids_list = []
        promotion_candidates = []
        
        from src.cubo.config import config as _config
        threshold = int(_config.get('vector_index.promote_threshold', 10))
        
        for res in results:
            did = res['id']
            docs.append(self._docs.get(did, ''))
            metas.append(self._metas.get(did, {}))
            dists.append(res['distance'])
            ids_list.append(did)
            # track access counts to potentially promote to hot
            self._access_counts[did] = self._access_counts.get(did, 0) + 1
            if self._access_counts[did] >= threshold:
                promotion_candidates.append(did)
        
        # Queue promotions asynchronously (non-blocking)
        if promotion_candidates:
            self._queue_promotions(promotion_candidates)
        
        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids_list]}

    def _queue_promotions(self, doc_ids: List[str]) -> None:
        """Queue doc IDs for async promotion to hot index.
        
        Promotions are batched and executed in a background thread to avoid
        blocking query operations. If a promotion is already in progress,
        new candidates are queued for the next batch.
        
        Args:
            doc_ids: List of document IDs to promote
        """
        with self._promotion_lock:
            # Add to pending set (deduplicates automatically)
            for did in doc_ids:
                if did in self._embeddings and did not in self._pending_promotions:
                    self._pending_promotions.add(did)
            
            # If promotion already running, it will pick up pending items
            if self._promotion_in_progress:
                return
            
            # Start async promotion if there are pending items
            if self._pending_promotions:
                self._promotion_in_progress = True
                executor = _get_promotion_executor()
                executor.submit(self._run_async_promotion)

    def _run_async_promotion(self) -> None:
        """Background worker that processes pending promotions.
        
        This runs in a separate thread and batches all pending promotions
        into a single index rebuild for efficiency.
        """
        try:
            while True:
                # Grab current batch of pending promotions
                with self._promotion_lock:
                    if not self._pending_promotions:
                        self._promotion_in_progress = False
                        return
                    batch = list(self._pending_promotions)
                    self._pending_promotions.clear()
                
                # Perform the actual promotion (may take time)
                self._promote_batch_to_hot(batch)
                
        except Exception as e:
            # Log but don't crash - promotions are best-effort
            import logging
            logging.getLogger(__name__).debug(f"Async promotion error: {e}")
        finally:
            with self._promotion_lock:
                self._promotion_in_progress = False

    def _promote_batch_to_hot(self, doc_ids: List[str]) -> None:
        """Promote a batch of documents to hot index.
        
        Args:
            doc_ids: List of document IDs to promote
        """
        if not doc_ids:
            return
        
        # Filter to only valid IDs that exist in embeddings
        valid_ids = [did for did in doc_ids if did in self._embeddings]
        if not valid_ids:
            return
        
        # Rebuild index with promoted docs at the front (ensures they're in hot set)
        all_ids = list(self._embeddings.keys())
        for did in valid_ids:
            if did in all_ids:
                all_ids.remove(did)
        new_ids = valid_ids + all_ids
        
        vectors = [self._embeddings[did] for did in new_ids]
        self._index.build_indexes(vectors, new_ids, append=False)
        
        # Reset access counts for promoted docs
        for did in valid_ids:
            self._access_counts[did] = 0

    def promote_to_hot(self, doc_id: str) -> None:
        """Promote a cold doc into the hot set (sync version for backward compatibility).
        
        For non-blocking promotion, use _queue_promotions() instead.
        This method is kept for API compatibility but internally queues
        the promotion asynchronously.

        Args:
            doc_id: Document ID to promote
        """
        self._queue_promotions([doc_id])

    def promote_to_hot_sync(self, doc_id: str) -> None:
        """Synchronously promote a doc to hot index (blocks until complete).
        
        Use this only when you need guaranteed immediate promotion.
        For normal use, prefer promote_to_hot() which is non-blocking.

        Args:
            doc_id: Document ID to promote
        """
        if doc_id not in self._embeddings:
            return
        self._promote_batch_to_hot([doc_id])

    def save(self, path: Optional[Path] = None) -> None:
        self._index.save(path)

    def load(self, path: Optional[Path] = None) -> None:
        self._index.load(path)

    def reset(self) -> None:
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        self._index = FAISSIndexManager(self.dimension, index_dir=self.index_dir, index_root=getattr(self, 'index_root', None))
        self._index.hot_fraction = self.hot_fraction
        self._docs.clear()
        self._metas.clear()
        self._embeddings.clear()
        self._access_counts.clear()

    def delete(self, ids=None) -> None:
        """Delete entries from the FAISS store by removing them from internal maps and rebuilding the index."""
        if not ids:
            return
        id_set = set(ids)
        # remove from local metadata
        for did in list(self._docs.keys()):
            if did in id_set:
                self._docs.pop(did, None)
                self._metas.pop(did, None)
                self._embeddings.pop(did, None)
                self._access_counts.pop(did, None)
        # Rebuild indexes with remaining data
        remaining_ids = list(self._embeddings.keys())
        vectors = [self._embeddings[did] for did in remaining_ids]
        try:
            self._index.build_indexes(vectors, remaining_ids, append=False)
        except Exception:
            # If rebuild fails, reset the index and fallback to empty
            self.reset()


def create_vector_store(backend: str = None, collection_name: Optional[str] = None, **kwargs) -> VectorStore:
    """Create a FAISS vector store instance.
    
    Args:
        backend: Legacy parameter, ignored (FAISS is always used)
        collection_name: Not used for FAISS
        **kwargs: Additional arguments for FaissStore (dimension, index_dir, index_root)
    
    Returns:
        FaissStore instance
    """
    dimension = kwargs.get('dimension', 1536)
    index_dir_arg = kwargs.get('index_dir', config.get('vector_store_path'))
    index_dir = Path(index_dir_arg) if index_dir_arg else None
    index_root_arg = kwargs.get('index_root', config.get('faiss_index_root', None))
    index_root = Path(index_root_arg) if index_root_arg else None
    return FaissStore(dimension, index_dir=index_dir, index_root=index_root)
