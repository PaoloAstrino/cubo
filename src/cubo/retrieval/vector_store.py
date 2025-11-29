"""
Vector store abstraction with FAISS as the primary backend.

Resource Optimization:
- Document text and metadata are stored in SQLite instead of in-memory dicts
- An LRU cache provides fast access to frequently-queried documents
- Embeddings are kept in memory for index rebuilds but can be persisted
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.cubo.config import config
from src.cubo.utils.trace_collector import trace_collector


class DocumentCache:
    """Thread-safe LRU cache for document content and metadata.

    Reduces SQLite reads for frequently accessed documents during queries.
    """

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document from cache, moving it to end (most recent)."""
        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
                self._hits += 1
                return self._cache[doc_id]
            self._misses += 1
            return None

    def put(self, doc_id: str, document: str, metadata: Dict) -> None:
        """Add or update document in cache."""
        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[doc_id] = {"document": document, "metadata": metadata}

    def remove(self, doc_id: str) -> None:
        """Remove document from cache."""
        with self._lock:
            self._cache.pop(doc_id, None)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total * 100) if total > 0 else 0,
            }


# Module-level executor for async operations (lazy initialized)
_promotion_executor: Optional[ThreadPoolExecutor] = None
_promotion_executor_users: int = 0
_promotion_lock = threading.Lock()


def _get_promotion_executor() -> ThreadPoolExecutor:
    """Get or create the shared ThreadPoolExecutor for async promotions."""
    # For backward compatibility, create executor if missing but do not change the
    # active-user reference counter (increment is intentionally not performed here).
    return start_promotion_executor(increment=False)


def start_promotion_executor(increment: bool = True) -> ThreadPoolExecutor:
    """Ensure a module-level promotion executor is running and return it.

    If ``increment`` is true, this call registers another user of the executor
    that must be balanced by a call to ``stop_promotion_executor`` to allow
    complete shutdown.
    """
    global _promotion_executor, _promotion_executor_users
    with _promotion_lock:
        if _promotion_executor is None:
            _promotion_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="faiss_promote"
            )
        if increment:
            _promotion_executor_users += 1
        return _promotion_executor


def stop_promotion_executor(force: bool = False) -> None:
    """Unregister a user of the promotion executor and shutdown if no users remain.

    If ``force`` is True, shutdown immediately regardless of active users.
    """
    global _promotion_executor, _promotion_executor_users
    with _promotion_lock:
        if force:
            try:
                if _promotion_executor is not None:
                    _promotion_executor.shutdown(wait=False)
            finally:
                _promotion_executor = None
                _promotion_executor_users = 0
            return

        if _promotion_executor_users > 0:
            _promotion_executor_users -= 1
        if _promotion_executor_users <= 0 and _promotion_executor is not None:
            try:
                _promotion_executor.shutdown(wait=False)
            except Exception:
                pass
            _promotion_executor = None
            _promotion_executor_users = 0


class VectorStore:
    def add(
        self,
        embeddings=None,
        documents=None,
        metadatas=None,
        ids=None,
        trace_id: Optional[str] = None,
    ):
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()

    def get(self, include=None, where=None, ids=None):
        raise NotImplementedError()

    def query(
        self,
        query_embeddings=None,
        n_results=10,
        include=None,
        where=None,
        trace_id: Optional[str] = None,
    ):
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

    Resource Optimization:
    - Document text and metadata stored in SQLite (not RAM)
    - LRU cache for frequently accessed documents
    - Embeddings kept in memory for index rebuilds
    """

    def __init__(
        self, dimension: int, index_dir: Optional[Path] = None, index_root: Optional[Path] = None
    ):
        self.dimension = dimension
        self.index_dir = (
            Path(index_dir) if index_dir else Path(config.get("vector_store_path", "./faiss_store"))
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)

        from src.cubo.indexing.faiss_index import FAISSIndexManager

        self._index = FAISSIndexManager(dimension, index_dir=self.index_dir, index_root=index_root)
        self.index_root = index_root

        # SQLite-backed document storage (replaces in-memory _docs/_metas)
        self._db_path = self.index_dir / "documents.db"
        self._init_document_db()

        # LRU cache for fast access to frequently-queried documents
        cache_size = int(config.get("document_cache_size", 1000))
        self._doc_cache = DocumentCache(max_size=cache_size)

        # Embeddings kept in memory for index rebuilds (needed for hot/cold promotion)
        self._embeddings: Dict[str, List[float]] = {}
        self._access_counts: Dict[str, int] = {}

        # Load embeddings from disk if available
        self._load_embeddings_from_disk()

        # Try to load existing FAISS indexes if available
        indexes_loaded = False
        try:
            self._index.load()
            from src.cubo.utils.logger import logger

            logger.info(
                f"Loaded existing FAISS indexes with {len(self._index.hot_ids)} hot and {len(self._index.cold_ids)} cold vectors"
            )
            indexes_loaded = True
        except FileNotFoundError:
            pass  # No existing indexes
        except Exception as e:
            from src.cubo.utils.logger import logger

            logger.warning(f"Failed to load existing FAISS indexes: {e}")

        # If indexes not loaded but we have embeddings, rebuild them
        if not indexes_loaded and self._embeddings:
            from src.cubo.utils.logger import logger

            logger.info(f"Rebuilding FAISS indexes from {len(self._embeddings)} stored embeddings")
            try:
                ids = list(self._embeddings.keys())
                vectors = [self._embeddings[did] for did in ids]
                self._index.build_indexes(vectors, ids, append=False)
                self._index.save()
                logger.info(f"Rebuilt and saved FAISS indexes with {len(ids)} vectors")
            except Exception as e:
                logger.warning(f"Failed to rebuild FAISS indexes: {e}")

        # Configure hot fraction from config
        from src.cubo.config import config as _config

        self.hot_fraction = float(_config.get("vector_index.hot_ratio", 0.2))
        self._index.hot_fraction = self.hot_fraction

        # Async promotion state
        self._pending_promotions: Set[str] = set()
        self._promotion_lock = threading.Lock()
        self._promotion_in_progress = False
        # Register as an active user of the module-level executor so that
        # background threads are present while the instance exists.
        start_promotion_executor(increment=True)
        self._closed = False

        # LRU cache for fast access to frequently-queried documents
        cache_size = int(config.get("document_cache_size", 1000))
        self._doc_cache = DocumentCache(max_size=cache_size)

        # Embeddings kept in memory for index rebuilds (needed for hot/cold promotion)
        self._embeddings: Dict[str, List[float]] = {}
        self._access_counts: Dict[str, int] = {}

        # Load embeddings from disk if available
        self._load_embeddings_from_disk()

        # Configure hot fraction from config
        from src.cubo.config import config as _config

        self.hot_fraction = float(_config.get("vector_index.hot_ratio", 0.2))
        self._index.hot_fraction = self.hot_fraction

        # Async promotion state
        self._pending_promotions: Set[str] = set()
        self._promotion_lock = threading.Lock()
        self._promotion_in_progress = False

    def _init_document_db(self) -> None:
        """Initialize SQLite database for document storage and collections."""
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(id)")
            
            # Collections table for organizing documents into containers
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    color TEXT DEFAULT '#2563eb'
                )
            """
            )
            
            # Junction table linking documents to collections
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS collection_documents (
                    collection_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (collection_id, document_id),
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_coll_doc_coll ON collection_documents(collection_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_coll_doc_doc ON collection_documents(document_id)")
            conn.commit()

    def _load_embeddings_from_disk(self) -> None:
        """Load embeddings from disk if a cache file exists."""
        import numpy as np

        emb_path = self.index_dir / "embeddings.npz"
        if emb_path.exists():
            try:
                data = np.load(str(emb_path), allow_pickle=True)
                ids = data["ids"].tolist()
                vectors = data["vectors"].tolist()
                for i, doc_id in enumerate(ids):
                    self._embeddings[doc_id] = vectors[i]
                    self._access_counts.setdefault(doc_id, 0)
            except Exception:
                pass  # If loading fails, we'll rebuild embeddings on add

    def _save_embeddings_to_disk(self) -> None:
        """Save embeddings to disk for persistence."""
        import numpy as np

        if not self._embeddings:
            return
        emb_path = self.index_dir / "embeddings.npz"
        try:
            ids = list(self._embeddings.keys())
            vectors = [self._embeddings[did] for did in ids]
            np.savez(str(emb_path), ids=np.array(ids), vectors=np.array(vectors))
        except Exception:
            pass  # Best effort - don't crash if save fails

    def _get_document_from_db(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document from SQLite."""
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            row = conn.execute(
                "SELECT content, metadata FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row:
            return {"document": row[0], "metadata": json.loads(row[1])}
        return None

    def _get_documents_batch(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple documents from SQLite in a single query."""
        if not doc_ids:
            return {}

        results = {}
        # First check cache
        uncached_ids = []
        for doc_id in doc_ids:
            cached = self._doc_cache.get(doc_id)
            if cached:
                results[doc_id] = cached
            else:
                uncached_ids.append(doc_id)

        # Fetch uncached from DB
        if uncached_ids:
            placeholders = ",".join("?" * len(uncached_ids))
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                rows = conn.execute(
                    f"SELECT id, content, metadata FROM documents WHERE id IN ({placeholders})",
                    uncached_ids,
                ).fetchall()
            for row in rows:
                doc_id, content, metadata_json = row
                metadata = json.loads(metadata_json)
                results[doc_id] = {"document": content, "metadata": metadata}
                self._doc_cache.put(doc_id, content, metadata)

        return results

    def add(
        self,
        embeddings=None,
        documents=None,
        metadatas=None,
        ids=None,
        trace_id: Optional[str] = None,
    ):
        """Add documents to the store.

        Documents and metadata are stored in SQLite for memory efficiency.
        Embeddings are kept in memory for index rebuilds.
        """
        if not embeddings or not ids:
            return

        # Persist to FAISS index
        self._index.build_indexes(embeddings, ids, append=True)

        # Store embeddings in memory for potential rebuilds (e.g., promotion to hot)
        for i, did in enumerate(ids):
            self._embeddings[did] = embeddings[i]
            self._access_counts.setdefault(did, 0)

        # Store documents and metadata in SQLite (not RAM)
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            for i, did in enumerate(ids):
                doc = documents[i] if documents and i < len(documents) else ""
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                conn.execute(
                    "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                    (did, doc, json.dumps(meta)),
                )
                # Also update cache
                self._doc_cache.put(did, doc, meta)
            conn.commit()

        # Save FAISS indexes to disk for persistence
        self._index.save()

        # Periodically save embeddings to disk
        if len(self._embeddings) % 100 == 0:
            self._save_embeddings_to_disk()
        # Record vector add event optionally
        if trace_id:
            try:
                trace_collector.record(
                    trace_id,
                    "vector_store",
                    "vector.added",
                    {
                        "ids": ids,
                        "hot_count": (
                            len(self._index.hot_ids) if hasattr(self._index, "hot_ids") else 0
                        ),
                        "cold_count": (
                            len(self._index.cold_ids) if hasattr(self._index, "cold_ids") else 0
                        ),
                    },
                )
            except Exception:
                pass

    def count(self) -> int:
        """Return total number of documents in store."""
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def get(self, include=None, where=None, ids=None):
        """Retrieve documents by IDs or filter.

        Uses cache for frequently accessed documents, falls back to SQLite.
        """
        ids_out = []
        docs = []
        metas = []

        if ids:
            # Batch fetch by IDs
            results = self._get_documents_batch(ids)
            for did in ids:
                if did in results:
                    ids_out.append(did)
                    docs.append(results[did]["document"])
                    metas.append(results[did]["metadata"])
        else:
            # Fetch all or filter by where clause
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                rows = conn.execute("SELECT id, content, metadata FROM documents").fetchall()

            for row in rows:
                did, content, metadata_json = row
                meta = json.loads(metadata_json)

                # Apply where filter if provided
                if where and isinstance(where, dict):
                    match = True
                    for key, val in where.items():
                        if meta.get(key) != val:
                            match = False
                            break
                    if not match:
                        continue

                ids_out.append(did)
                docs.append(content)
                metas.append(meta)

        return {"ids": ids_out, "documents": [docs], "metadatas": [metas]}

    def query(
        self,
        query_embeddings=None,
        n_results=10,
        include=None,
        where=None,
        trace_id: Optional[str] = None,
    ):
        """Query FAISS for nearest neighbors with async hot promotion.

        Uses batch document fetching with cache for efficiency.
        """
        if not query_embeddings or self.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        results = self._index.search(query_embeddings[0], k=n_results)

        # Collect all doc IDs for batch fetch
        result_ids = [res["id"] for res in results]
        doc_data = self._get_documents_batch(result_ids)

        # Map results to documents/metas/ids/distance
        docs = []
        metas = []
        dists = []
        ids_list = []
        promotion_candidates = []

        from src.cubo.config import config as _config

        threshold = int(_config.get("vector_index.promote_threshold", 10))

        for res in results:
            did = res["id"]
            data = doc_data.get(did, {"document": "", "metadata": {}})
            docs.append(data["document"])
            metas.append(data["metadata"])
            dists.append(res["distance"])
            ids_list.append(did)

            # Track access counts to potentially promote to hot
            self._access_counts[did] = self._access_counts.get(did, 0) + 1
            if self._access_counts[did] >= threshold:
                promotion_candidates.append(did)

        # Queue promotions asynchronously (non-blocking)
        if promotion_candidates:
            self._queue_promotions(promotion_candidates)

        # Record query event: top ids/dists
        if trace_id:
            try:
                trace_collector.record(
                    trace_id,
                    "vector_store",
                    "vector.query",
                    {"ids": ids_list, "distances_sample": dists[:5]},
                )
            except Exception:
                pass

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
                # Ensure executor is present but do not increment user counter
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
        """Save index and embeddings to disk."""
        self._index.save(path)
        self._save_embeddings_to_disk()

    def load(self, path: Optional[Path] = None) -> None:
        """Load index from disk."""
        self._index.load(path)
        self._load_embeddings_from_disk()

    # =========================================================================
    # Collection Management Methods
    # =========================================================================

    def create_collection(self, name: str, color: str = "#2563eb") -> Dict[str, Any]:
        """Create a new document collection.
        
        Args:
            name: Unique name for the collection
            color: Hex color for visual representation (default: brand blue)
            
        Returns:
            Dict with collection id, name, color, and created_at
            
        Raises:
            ValueError: If collection name already exists
        """
        import uuid
        from datetime import datetime
        
        collection_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            try:
                conn.execute(
                    "INSERT INTO collections (id, name, created_at, color) VALUES (?, ?, ?, ?)",
                    (collection_id, name, created_at, color)
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Collection '{name}' already exists")
        
        return {
            "id": collection_id,
            "name": name,
            "color": color,
            "created_at": created_at,
            "document_count": 0
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with document counts.
        
        Returns:
            List of collection dicts with id, name, color, created_at, document_count
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.color, c.created_at,
                       COUNT(cd.document_id) as document_count
                FROM collections c
                LEFT JOIN collection_documents cd ON c.id = cd.collection_id
                GROUP BY c.id
                ORDER BY c.created_at DESC
                """
            ).fetchall()
            
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "created_at": row["created_at"],
                "document_count": row["document_count"]
            }
            for row in rows
        ]

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific collection by ID.
        
        Args:
            collection_id: The collection's unique ID
            
        Returns:
            Collection dict or None if not found
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT c.id, c.name, c.color, c.created_at,
                       COUNT(cd.document_id) as document_count
                FROM collections c
                LEFT JOIN collection_documents cd ON c.id = cd.collection_id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (collection_id,)
            ).fetchone()
            
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "created_at": row["created_at"],
                "document_count": row["document_count"]
            }
        return None

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection (documents remain in store, just unlinked).
        
        Args:
            collection_id: The collection's unique ID
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            # Delete from junction table first (foreign key cascade would do this too)
            conn.execute(
                "DELETE FROM collection_documents WHERE collection_id = ?",
                (collection_id,)
            )
            cursor = conn.execute(
                "DELETE FROM collections WHERE id = ?",
                (collection_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_documents_to_collection(
        self, collection_id: str, document_ids: List[str]
    ) -> Dict[str, Any]:
        """Add documents to a collection.
        
        Args:
            collection_id: The collection's unique ID
            document_ids: List of document IDs to add
            
        Returns:
            Dict with added_count and already_in_collection count
        """
        from datetime import datetime
        
        added_at = datetime.utcnow().isoformat()
        added_count = 0
        already_exists = 0
        
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            for doc_id in document_ids:
                try:
                    conn.execute(
                        "INSERT INTO collection_documents (collection_id, document_id, added_at) VALUES (?, ?, ?)",
                        (collection_id, doc_id, added_at)
                    )
                    added_count += 1
                except sqlite3.IntegrityError:
                    already_exists += 1
            conn.commit()
        
        return {"added_count": added_count, "already_in_collection": already_exists}

    def remove_documents_from_collection(
        self, collection_id: str, document_ids: List[str]
    ) -> int:
        """Remove documents from a collection.
        
        Args:
            collection_id: The collection's unique ID
            document_ids: List of document IDs to remove
            
        Returns:
            Number of documents removed
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            placeholders = ",".join("?" * len(document_ids))
            cursor = conn.execute(
                f"DELETE FROM collection_documents WHERE collection_id = ? AND document_id IN ({placeholders})",
                [collection_id] + list(document_ids)
            )
            conn.commit()
            return cursor.rowcount

    def get_collection_documents(self, collection_id: str) -> List[str]:
        """Get all document IDs in a collection.
        
        Args:
            collection_id: The collection's unique ID
            
        Returns:
            List of document IDs
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            rows = conn.execute(
                "SELECT document_id FROM collection_documents WHERE collection_id = ? ORDER BY added_at DESC",
                (collection_id,)
            ).fetchall()
        return [row[0] for row in rows]

    def get_document_filenames_in_collection(self, collection_id: str) -> List[str]:
        """Get all filenames of documents in a collection (for query filtering).
        
        Args:
            collection_id: The collection's unique ID
            
        Returns:
            List of unique filenames
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT d.metadata
                FROM documents d
                JOIN collection_documents cd ON d.id = cd.document_id
                WHERE cd.collection_id = ?
                """,
                (collection_id,)
            ).fetchall()
        
        filenames = set()
        for row in rows:
            try:
                metadata = json.loads(row[0])
                if "filename" in metadata:
                    filenames.add(metadata["filename"])
            except (json.JSONDecodeError, KeyError):
                pass
        
        return list(filenames)

    # =========================================================================
    # End Collection Management Methods
    # =========================================================================

    def reset(self) -> None:
        """Reset the store, clearing all data."""
        from src.cubo.indexing.faiss_index import FAISSIndexManager

        self._index = FAISSIndexManager(
            self.dimension, index_dir=self.index_dir, index_root=getattr(self, "index_root", None)
        )
        self._index.hot_fraction = self.hot_fraction

        # Clear SQLite database; ensure index dir exists and attempt retry in case of transient locks
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Ensure DB tables exist (init will create the documents table if missing)
        try:
            self._init_document_db()
        except Exception:
            pass

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                    conn.execute("DELETE FROM documents")
                    conn.commit()
                break
            except sqlite3.OperationalError as e:
                # If the table doesn't exist, nothing to clear - exit gracefully
                if "no such table" in str(e).lower():
                    break
                if attempt == max_attempts:
                    raise
                time.sleep(0.05 * attempt)

        # Clear caches
        self._doc_cache.clear()
        self._embeddings.clear()
        self._access_counts.clear()

        # Remove embeddings file
        emb_path = self.index_dir / "embeddings.npz"
        if emb_path.exists():
            try:
                os.remove(str(emb_path))
            except Exception:
                pass

        # Note: we do not shutdown the module-level executor here; use close()
        # to properly unregister and shutdown when an instance is destroyed.

    def delete(self, ids=None) -> None:
        """Delete entries from the FAISS store."""
        if not ids:
            return
        id_set = set(ids)

        # Remove from SQLite
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", list(ids))
            conn.commit()

        # Remove from cache and in-memory stores
        for did in ids:
            self._doc_cache.remove(did)
            self._embeddings.pop(did, None)
            self._access_counts.pop(did, None)

        # Rebuild indexes with remaining data
        remaining_ids = list(self._embeddings.keys())
        if remaining_ids:
            vectors = [self._embeddings[did] for did in remaining_ids]
            try:
                self._index.build_indexes(vectors, remaining_ids, append=False)
            except Exception:
                # If rebuild fails, reset the index and fallback to empty
                self.reset()
        else:
            self.reset()

    def close(self, persist: bool = False) -> None:
        """Shutdown resources used by this FaissStore instance.

        This will unregister the instance from the module-level promotion
        executor, and optionally persist the index files.
        """
        if getattr(self, "_closed", False):
            return
        try:
            if persist:
                try:
                    self._index.save()
                except Exception:
                    pass
            # Clear caches
            self._doc_cache.clear()
            self._embeddings.clear()
            self._access_counts.clear()
            # Release index references so GC can collect them
            try:
                self._index = None  # type: ignore
            except Exception:
                pass
        finally:
            # Unregister from the module-level executor and potentially shut it down
            try:
                stop_promotion_executor()
            except Exception:
                pass
            self._closed = True

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return document cache statistics for monitoring."""
        return self._doc_cache.stats


def create_vector_store(
    backend: str = None, collection_name: Optional[str] = None, **kwargs
) -> VectorStore:
    """Create a FAISS vector store instance.

    Args:
        backend: Legacy parameter, ignored (FAISS is always used)
        collection_name: Not used for FAISS
        **kwargs: Additional arguments for FaissStore (dimension, index_dir, index_root)

    Returns:
        FaissStore instance
    """
    dimension = kwargs.get("dimension", 1536)
    index_dir_arg = kwargs.get("index_dir", config.get("vector_store_path"))
    index_dir = Path(index_dir_arg) if index_dir_arg else None
    index_root_arg = kwargs.get("index_root", config.get("faiss_index_root", None))
    index_root = Path(index_root_arg) if index_root_arg else None
    return FaissStore(dimension, index_dir=index_dir, index_root=index_root)
