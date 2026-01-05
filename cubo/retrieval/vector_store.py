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
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from cubo.config import config
from cubo.utils.logger import logger
from cubo.utils.trace_collector import trace_collector

# Promotion throttling constants to prevent RAM spikes
MIN_REBUILD_INTERVAL_SECONDS = 60  # Minimum time between index rebuilds
MAX_PROMOTIONS_PER_REBUILD = 50  # Maximum docs to promote per rebuild cycle


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
        if self._max_size <= 0:
            return

        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
            else:
                while len(self._cache) >= self._max_size and self._cache:
                    self._cache.popitem(last=False)

            if len(self._cache) < self._max_size:
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


class VectorCache:
    """Thread-safe LRU cache for vector embeddings.

    Reduces SQLite reads for frequently accessed vectors during index building
    and promotion.
    """

    def __init__(self, max_size: int = 5000):
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, doc_id: str) -> Optional[Any]:
        """Get vector from cache, moving it to end (most recent)."""
        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
                self._hits += 1
                return self._cache[doc_id]
            self._misses += 1
            return None

    def put(self, doc_id: str, vector: Any) -> None:
        """Add or update vector in cache."""
        if self._max_size <= 0:
            return

        with self._lock:
            if doc_id in self._cache:
                self._cache.move_to_end(doc_id)
            else:
                while len(self._cache) >= self._max_size and self._cache:
                    self._cache.popitem(last=False)

            if len(self._cache) < self._max_size:
                self._cache[doc_id] = vector

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
    ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

    def _validate_ids(self, ids: List[str]) -> None:
        """Validate a list of ids to ensure they contain only safe characters.

        Raises ValueError if any id contains unexpected characters which could
        be used to attempt SQL injection via crafted IDs.
        """
        if not ids:
            return
        for i in ids:
            if not isinstance(i, (str, bytes)):
                raise ValueError("Invalid id type; expected str or bytes")
            s = i.decode() if isinstance(i, bytes) else i
            if not self.ID_RE.match(s):
                raise ValueError(f"Invalid id value detected: {s}")

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

        self._write_lock = threading.RLock()  # Protects against concurrent SQLite writes

        from cubo.indexing.faiss_index import FAISSIndexManager

        self._index = FAISSIndexManager(dimension, index_dir=self.index_dir, index_root=index_root)
        self.index_root = index_root

        # SQLite-backed document storage (replaces in-memory _docs/_metas)
        self._db_path = self.index_dir / "documents.db"
        self._init_document_db()

        # LRU cache for fast access to frequently-queried documents
        cache_size = int(config.get("document_cache_size", 1000))
        self._doc_cache = DocumentCache(max_size=cache_size)

        # LRU cache for vectors (replaces full in-memory dict)
        vector_cache_size = int(config.get("vector_store.cache_size", 5000))
        self._vector_cache = VectorCache(max_size=vector_cache_size)

        # Embeddings are now fetched from DB on demand
        self._access_counts: Dict[str, int] = {}

        # Try to load existing FAISS indexes if available
        indexes_loaded = False
        try:
            self._index.load()
            from cubo.utils.logger import logger

            logger.info(
                f"Loaded existing FAISS indexes with {len(self._index.hot_ids)} hot and {len(self._index.cold_ids)} cold vectors"
            )
            indexes_loaded = True
        except FileNotFoundError:
            pass  # No existing indexes
        except Exception as e:
            from cubo.utils.logger import logger

            logger.warning(f"Failed to load existing FAISS indexes: {e}")

        # If indexes not loaded, we might need to rebuild from DB
        if not indexes_loaded:
            count = self.count_vectors()
            if count > 0:
                from cubo.utils.logger import logger

                logger.info(f"Rebuilding FAISS indexes from {count} stored vectors in DB")
                try:
                    self._rebuild_index_from_db()
                    logger.info("Rebuilt and saved FAISS indexes")
                except Exception as e:
                    logger.warning(f"Failed to rebuild FAISS indexes: {e}")

        # Configure hot fraction from config
        from cubo.config import config as _config

        self.hot_fraction = float(_config.get("vector_index.hot_ratio", 0.2))
        self._index.hot_fraction = self.hot_fraction

        # Async promotion state
        self._pending_promotions: Set[str] = set()
        self._promotion_lock = threading.Lock()
        self._promotion_in_progress = False
        self._last_rebuild_time: float = 0.0  # For throttling
        # Register as an active user of the module-level executor so that
        # background threads are present while the instance exists.
        start_promotion_executor(increment=True)
        self._closed = False

        # Migrate old embeddings.npz to SQLite if needed
        self._migrate_embeddings_if_needed()

    def _init_document_db(self) -> None:
        """Initialize SQLite database for document storage and collections."""
        with self._write_lock:  # Protect init
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coll_doc_coll ON collection_documents(collection_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_coll_doc_doc ON collection_documents(document_id)"
                )

                # Vectors table for storing embeddings
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vectors (
                        id TEXT PRIMARY KEY,
                        vector BLOB NOT NULL,
                        dtype TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """
                )

                # Jobs enqueued for background deletions/compaction
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS deletion_jobs (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        enqueued_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        priority INTEGER DEFAULT 0,
                        trace_id TEXT,
                        force INTEGER DEFAULT 0
                    )
                """
                )
                conn.commit()

    def _serialize_vector(self, vector: Any) -> tuple[bytes, str, int]:
        """Serialize numpy vector to bytes."""
        import numpy as np

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)

        return vector.tobytes(), str(vector.dtype), vector.shape[0]

    def _deserialize_vector(self, blob: bytes, dtype: str, dim: int) -> Any:
        """Deserialize bytes to numpy vector."""
        import numpy as np

        return np.frombuffer(blob, dtype=dtype).reshape(dim)

    def _migrate_embeddings_if_needed(self) -> None:
        """Migrate embeddings from .npz to SQLite if needed."""

        import numpy as np

        emb_path = self.index_dir / "embeddings.npz"
        if not emb_path.exists():
            return

        # Check if we already have vectors in DB
        if self.count_vectors() > 0:
            return

        from cubo.utils.logger import logger

        logger.info("Migrating embeddings from .npz to SQLite...")

        try:
            data = np.load(str(emb_path), allow_pickle=True)
            ids = data["ids"].tolist()
            vectors = data["vectors"].tolist()

            self.save_vectors(ids, vectors)

            # Rename .npz to indicate it's migrated
            emb_path.rename(emb_path.with_suffix(".npz.migrated"))
            logger.info(f"Successfully migrated {len(ids)} vectors to SQLite")

        except Exception as e:
            logger.error(f"Failed to migrate embeddings: {e}")

    def save_vectors(self, ids: List[str], vectors: List[Any]) -> None:
        """Save vectors to SQLite."""
        from datetime import datetime

        created_at = datetime.utcnow().isoformat()

        with self._write_lock:  # Protect write
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

                data = []
                for doc_id, vector in zip(ids, vectors):
                    blob, dtype, dim = self._serialize_vector(vector)
                    data.append((doc_id, blob, dtype, dim, created_at))
                    # Update cache
                    self._vector_cache.put(doc_id, vector)

                conn.executemany(
                    "INSERT OR REPLACE INTO vectors (id, vector, dtype, dim, created_at) VALUES (?, ?, ?, ?, ?)",
                    data,
                )
                conn.commit()

    def get_vector(self, doc_id: str) -> Optional[Any]:
        """Get a single vector by ID."""
        # Check cache first
        cached = self._vector_cache.get(doc_id)
        if cached is not None:
            return cached

        # Check FAISS index reconstruction (fastest after cache)
        if self._index:
            reconstructed = self._index.get_vector(doc_id)
            if reconstructed is not None:
                self._vector_cache.put(doc_id, reconstructed)
                return reconstructed

        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            row = conn.execute(
                "SELECT vector, dtype, dim FROM vectors WHERE id = ?", (doc_id,)
            ).fetchone()

        if row:
            vector = self._deserialize_vector(row[0], row[1], row[2])
            self._vector_cache.put(doc_id, vector)
            return vector
        return None

    def get_vectors(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Get multiple vectors by IDs."""
        results = {}
        uncached_ids = []

        # Check cache
        for doc_id in doc_ids:
            cached = self._vector_cache.get(doc_id)
            if cached is not None:
                results[doc_id] = cached
            else:
                # Try reconstruction
                if self._index:
                    reconstructed = self._index.get_vector(doc_id)
                    if reconstructed is not None:
                        results[doc_id] = reconstructed
                        self._vector_cache.put(doc_id, reconstructed)
                        continue
                uncached_ids.append(doc_id)

        if not uncached_ids:
            return results

        # Fetch uncached from DB
        # Chunk large requests to avoid SQL limits
        chunk_size = 900  # SQLite limit is usually 999 vars
        for i in range(0, len(uncached_ids), chunk_size):
            chunk = uncached_ids[i : i + chunk_size]

            # Validate IDs to prevent injection via unexpected characters
            self._validate_ids(chunk)

            # Use a temporary table to avoid dynamic SQL in IN() clause and prevent injection
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                cur = conn.cursor()
                cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ids(id TEXT PRIMARY KEY)")
                try:
                    cur.executemany(
                        "INSERT OR REPLACE INTO _tmp_ids(id) VALUES(?)",
                        ((i,) for i in chunk),
                    )
                    cur.execute(
                        "SELECT id, vector, dtype, dim FROM vectors WHERE id IN (SELECT id FROM _tmp_ids)"
                    )
                    rows = cur.fetchall()
                finally:
                    cur.execute("DELETE FROM _tmp_ids")
                    conn.commit()

            for row in rows:
                doc_id = row[0]
                vector = self._deserialize_vector(row[1], row[2], row[3])
                results[doc_id] = vector
                self._vector_cache.put(doc_id, vector)

        return results

    def count_vectors(self) -> int:
        """Count total vectors in DB."""
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return row[0] if row else 0

    def _rebuild_index_from_db(self) -> None:
        """Rebuild FAISS index using streaming to prevent OOM."""
        total = self.count_vectors()
        if total == 0:
            return

        # Phase 1: Training (if needed, use sample of up to 50k)
        logger.info("Starting streaming FAISS rebuild...")

        sample_size = 50000
        training_vectors = []

        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
             # Use parameterized LIMIT to avoid dynamic SQL construction flagged by security scanners
             cursor = conn.execute("SELECT id FROM vectors LIMIT ?", (int(sample_size),))
             sample_ids = [r[0] for r in cursor]

        if sample_ids:
            logger.info(f"Fetching training sample ({len(sample_ids)} vectors)...")
            batch_vecs = self.get_vectors(sample_ids)
            for did in sample_ids:
                if did in batch_vecs:
                    training_vectors.append(batch_vecs[did])

            # Reset and Train
            self._index.reset()
            self._index.train(training_vectors)

        # Phase 2: Streaming Add
        hot_capacity = int(total * self._index.hot_fraction)
        batch_size = 5000

        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            rows = conn.execute("SELECT id FROM vectors").fetchall()
            all_ids = [r[0] for r in rows]

        logger.info(f"Adding {len(all_ids)} vectors in batches of {batch_size}...")

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]
            batch_vecs_map = self.get_vectors(batch_ids)

            # Reconstruct batch lists in order
            batch_vectors_list = []
            batch_ids_list = []

            for did in batch_ids:
                if did in batch_vecs_map:
                    batch_vectors_list.append(batch_vecs_map[did])
                    batch_ids_list.append(did)

            if not batch_vectors_list:
                continue

            # Determine destination based on index position
            current_start = i
            remaining_hot = max(0, hot_capacity - current_start)

            if remaining_hot >= len(batch_vectors_list):
                # All hot
                self._index.add_batch(batch_vectors_list, batch_ids_list, destination="hot")
            elif remaining_hot <= 0:
                # All cold
                self._index.add_batch(batch_vectors_list, batch_ids_list, destination="cold")
            else:
                # Split
                hot_v = batch_vectors_list[:remaining_hot]
                hot_i = batch_ids_list[:remaining_hot]
                cold_v = batch_vectors_list[remaining_hot:]
                cold_i = batch_ids_list[remaining_hot:]
                self._index.add_batch(hot_v, hot_i, destination="hot")
                self._index.add_batch(cold_v, cold_i, destination="cold")

            if i % (batch_size * 5) == 0:
                logger.info(f"Processed {i}/{total} vectors...")

        self._index.save()
        logger.info("FAISS index rebuild complete.")

    def _load_embeddings_from_disk(self) -> None:
        """Deprecated: Use _migrate_embeddings_if_needed instead."""
        pass

    def _save_embeddings_to_disk(self) -> None:
        """Deprecated: Vectors are saved to SQLite immediately."""
        pass

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
            # Use temporary table to safely pass many IDs without building dynamic SQL
            self._validate_ids(uncached_ids)
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                cur = conn.cursor()
                cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ids(id TEXT PRIMARY KEY)")
                try:
                    cur.executemany(
                        "INSERT OR REPLACE INTO _tmp_ids(id) VALUES(?)",
                        ((i,) for i in uncached_ids),
                    )
                    cur.execute(
                        "SELECT id, content, metadata FROM documents WHERE id IN (SELECT id FROM _tmp_ids)"
                    )
                    rows = cur.fetchall()
                finally:
                    cur.execute("DELETE FROM _tmp_ids")
                    conn.commit()
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
        Uses two-phase commit: FAISS saves first, then SQLite commits.
        This prevents zombie documents (in SQLite but not in FAISS).
        """
        if not embeddings or not ids:
            return

        with self._write_lock:  # Protect add
            # Phase 1: Build FAISS indexes (in-memory)
            self._index.build_indexes(embeddings, ids, append=True)

            # Phase 2: Save FAISS indexes to disk BEFORE committing to SQLite
            # If this fails, we haven't touched SQLite yet
            try:
                self._index.save()
            except Exception as e:
                from cubo.utils.logger import logger

                logger.error(f"FAISS save failed, rolling back: {e}")
                # Rollback: remove from FAISS index
                # (rebuild without the new IDs would be expensive, so just log)
                raise

            # Phase 3: Now commit to SQLite (FAISS is safely persisted)
            conn = sqlite3.connect(str(self._db_path), timeout=30)
            try:
                # Store vectors
                from datetime import datetime

                created_at = datetime.utcnow().isoformat()
                for doc_id, vector in zip(ids, embeddings):
                    blob, dtype, dim = self._serialize_vector(vector)
                    conn.execute(
                        "INSERT OR REPLACE INTO vectors (id, vector, dtype, dim, created_at) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, blob, dtype, dim, created_at),
                    )
                    # Update cache
                    vec_np = np.asarray(vector, dtype=np.float32)
                    self._vector_cache.put(doc_id, vec_np)

                # Store documents and metadata
                for i, did in enumerate(ids):
                    doc = documents[i] if documents and i < len(documents) else ""
                    meta = metadatas[i] if metadatas and i < len(metadatas) else {}

                    # Prepare metadata for JSON storage: convert numpy arrays/ndarrays to lists
                    def _prepare(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, (np.floating, np.integer, np.bool_)):
                            return obj.item()
                        if isinstance(obj, dict):
                            return {k: _prepare(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [_prepare(x) for x in obj]
                        return obj

                    meta_jsonable = _prepare(meta)
                    conn.execute(
                        "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                        (did, doc, json.dumps(meta_jsonable)),
                    )
                    # Update cache
                    self._doc_cache.put(did, doc, meta)

                # Only commit after all inserts succeed
                conn.commit()

            except Exception as e:
                conn.rollback()
                from cubo.utils.logger import logger

                logger.error(f"SQLite commit failed after FAISS save: {e}")
                # Note: FAISS is already saved - this is a partial state
                # but next startup will rebuild from SQLite if needed
                raise
            finally:
                conn.close()

        # Init access counts
        for did in ids:
            self._access_counts.setdefault(did, 0)

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

                # Apply where filter if provided. Supports simple equality and
                # a minimal subset of operators such as $in for membership.
                if where and isinstance(where, dict):
                    match = True
                    for key, val in where.items():
                        if isinstance(val, dict) and "$in" in val:
                            # membership operator
                            if meta.get(key) not in val["$in"]:
                                match = False
                                break
                        else:
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

        from cubo.config import config as _config

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

        # Apply where filter at the result level too (if provided by caller).
        if where and isinstance(where, dict):
            filtered_docs, filtered_metas, filtered_dists, filtered_ids = [], [], [], []
            for doc, meta, dist, did in zip(docs, metas, dists, ids_list):
                match = True
                for key, val in where.items():
                    if isinstance(val, dict) and "$in" in val:
                        if meta.get(key) not in val["$in"]:
                            match = False
                            break
                    else:
                        if meta.get(key) != val:
                            match = False
                            break
                if match:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_dists.append(dist)
                    filtered_ids.append(did)
            docs, metas, dists, ids_list = (
                filtered_docs,
                filtered_metas,
                filtered_dists,
                filtered_ids,
            )

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
            # Add to pending set (deduplicates automatically)
            for did in doc_ids:
                # Check if vector exists in DB (or cache)
                if did not in self._pending_promotions:
                    # We can't easily check existence without query, but promotion is safe to fail
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
        into a single index rebuild for efficiency. Throttling prevents
        excessive rebuilds from causing RAM spikes.
        """
        try:
            while True:
                # Check minimum interval between rebuilds
                now = time.time()
                elapsed = now - self._last_rebuild_time
                if elapsed < MIN_REBUILD_INTERVAL_SECONDS:
                    wait_time = MIN_REBUILD_INTERVAL_SECONDS - elapsed
                    time.sleep(wait_time)

                # Grab current batch of pending promotions (limited)
                with self._promotion_lock:
                    if not self._pending_promotions:
                        self._promotion_in_progress = False
                        return
                    batch = list(self._pending_promotions)[:MAX_PROMOTIONS_PER_REBUILD]
                    # Remove processed items from pending set
                    for doc_id in batch:
                        self._pending_promotions.discard(doc_id)

                # Perform the actual promotion (may take time)
                self._promote_batch_to_hot(batch)
                self._last_rebuild_time = time.time()

        except Exception as e:
            # Log but don't crash - promotions are best-effort
            import logging

            logging.getLogger(__name__).debug(f"Async promotion error: {e}")
        finally:
            with self._promotion_lock:
                self._promotion_in_progress = False

    def _promote_batch_to_hot(self, doc_ids: List[str]) -> None:
        """Promote a batch of documents to hot index incrementally.

        This method uses incremental addition to the hot index and relies on
        search-time deduplication to handle overlap with the cold index.
        This avoids rebuilding the entire index (RAM bottleneck).
        """
        if not doc_ids:
            return

        # 1. Fetch vectors (will use cache/reconstruct/DB)
        vectors_map = self.get_vectors(doc_ids)

        valid_ids = []
        valid_vectors = []

        for did in doc_ids:
            if did in vectors_map:
                valid_ids.append(did)
                valid_vectors.append(vectors_map[did])

        if not valid_ids:
            return

        # 2. Add to hot index incrementally
        # This avoids the massive rebuild of the entire index
        self._index.add_to_hot(valid_vectors, valid_ids)

        # 3. Reset access counts for promoted docs
        for did in valid_ids:
            self._access_counts[did] = 0

        from cubo.utils.logger import logger

        logger.info(f"Promoted {len(valid_ids)} docs to hot index incrementally")

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
        if self.get_vector(doc_id) is None:
            return
        self._promote_batch_to_hot([doc_id])

    def save(self, path: Optional[Path] = None) -> None:
        """Save index and embeddings to disk."""
        self._index.save(path)
        # self._save_embeddings_to_disk() # No longer needed

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
                    (collection_id, name, created_at, color),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"Collection '{name}' already exists")

        return {
            "id": collection_id,
            "name": name,
            "color": color,
            "created_at": created_at,
            "document_count": 0,
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
                "document_count": row["document_count"],
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
                (collection_id,),
            ).fetchone()

        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "color": row["color"],
                "created_at": row["created_at"],
                "document_count": row["document_count"],
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
                "DELETE FROM collection_documents WHERE collection_id = ?", (collection_id,)
            )
            cursor = conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
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
                        (collection_id, doc_id, added_at),
                    )
                    added_count += 1
                except sqlite3.IntegrityError:
                    already_exists += 1
            conn.commit()

        return {"added_count": added_count, "already_in_collection": already_exists}

    def remove_documents_from_collection(self, collection_id: str, document_ids: List[str]) -> int:
        """Remove documents from a collection.

        Args:
            collection_id: The collection's unique ID
            document_ids: List of document IDs to remove

        Returns:
            Number of documents removed
        """
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            # Use temporary table to safely delete multiple document ids
            self._validate_ids(document_ids)
            cur = conn.cursor()
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ids(id TEXT PRIMARY KEY)")
            try:
                cur.executemany(
                    "INSERT OR REPLACE INTO _tmp_ids(id) VALUES(?)", ((i,) for i in document_ids)
                )
                cur.execute(
                    "DELETE FROM collection_documents WHERE collection_id = ? AND document_id IN (SELECT id FROM _tmp_ids)",
                    (collection_id,),
                )
                rowcount = cur.rowcount
                conn.commit()
                return rowcount
            finally:
                cur.execute("DELETE FROM _tmp_ids")
                conn.commit()

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
                (collection_id,),
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
                (collection_id,),
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
        from cubo.indexing.faiss_index import FAISSIndexManager

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
                    conn.execute("DELETE FROM vectors")
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
        self._vector_cache.clear()
        self._access_counts.clear()

        # Remove embeddings file (legacy)
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
        _id_set = set(ids)

        # Remove from SQLite (documents and vectors)
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            # Use a temporary table to safely delete many ids without building dynamic SQL
            self._validate_ids(ids)
            cur = conn.cursor()
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ids(id TEXT PRIMARY KEY)")
            try:
                cur.executemany(
                    "INSERT OR REPLACE INTO _tmp_ids(id) VALUES(?)", ((i,) for i in ids)
                )
                cur.execute("DELETE FROM documents WHERE id IN (SELECT id FROM _tmp_ids)")
                cur.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM _tmp_ids)")
                conn.commit()
            finally:
                cur.execute("DELETE FROM _tmp_ids")
                conn.commit()

        # Remove from cache and in-memory stores
        for did in ids:
            self._doc_cache.remove(did)
            self._access_counts.pop(did, None)

        # Rebuild indexes with remaining data from DB
        remaining_count = self.count_vectors()
        if remaining_count > 0:
            try:
                self._rebuild_index_from_db()
            except Exception:
                # If rebuild fails, reset the index and fallback to empty
                self.reset()
        else:
            self.reset()

    def enqueue_deletion(
        self, doc_id: str, trace_id: Optional[str] = None, force: bool = False
    ) -> str:
        """Enqueue a deletion job for background compaction.

        This method deletes the document and vector rows from SQLite immediately
        (logical/DB deletion) and enqueues a compaction job to rebuild FAISS
        index in the background. Returns a job id to track progress.
        """
        import uuid
        from datetime import datetime

        job_id = uuid.uuid4().hex[:8]
        enqueued_at = datetime.utcnow().isoformat()
        status = "pending"
        priority = 1 if force else 0

        with self._write_lock:
            # Validate id
            self._validate_ids([doc_id])
            # delete from documents/vectors immediately so DB reflects removal
            with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                cur = conn.cursor()
                try:
                    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    cur.execute("DELETE FROM vectors WHERE id = ?", (doc_id,))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

                # Insert job record
                cur.execute(
                    "INSERT OR REPLACE INTO deletion_jobs (id, doc_id, enqueued_at, status, priority, trace_id, force) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (job_id, doc_id, enqueued_at, status, priority, trace_id, 1 if force else 0),
                )
                conn.commit()

        # Clean caches
        self._doc_cache.remove(doc_id)
        self._access_counts.pop(doc_id, None)
        try:
            # Overwrite any cached vector reference
            self._vector_cache.put(doc_id, None)
        except Exception:
            pass

        # Schedule background compaction via promotion executor
        try:
            executor = _get_promotion_executor()
            executor.submit(self._run_compaction_once)
        except Exception:
            # If scheduling fails, leave job pending for manual/cron compaction
            pass

        return job_id

    def get_deletion_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return status for a deletion job."""
        with sqlite3.connect(str(self._db_path), timeout=30) as conn:
            row = conn.execute(
                "SELECT id, doc_id, enqueued_at, status, priority, trace_id, force FROM deletion_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "doc_id": row[1],
            "enqueued_at": row[2],
            "status": row[3],
            "priority": row[4],
            "trace_id": row[5],
            "force": bool(row[6]),
        }

    def _run_compaction_once(self) -> None:
        """Run a single compaction pass: pick pending jobs, rebuild index, mark jobs done."""
        from datetime import datetime

        with self._write_lock:
            try:
                with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                    cur = conn.cursor()
                    # Pick pending jobs ordered by priority then time
                    rows = cur.execute(
                        "SELECT id, doc_id FROM deletion_jobs WHERE status = 'pending' ORDER BY priority DESC, enqueued_at ASC LIMIT 100"
                    ).fetchall()
                    job_ids = [r[0] for r in rows]
                    if not job_ids:
                        return
                    # Mark in-progress
                    cur.executemany(
                        "UPDATE deletion_jobs SET status = 'in_progress' WHERE id = ?",
                        ((jid,) for jid in job_ids),
                    )
                    conn.commit()

                # Perform rebuild from DB (which no longer contains deleted vectors)
                try:
                    self._rebuild_index_from_db()
                    success = True
                except Exception:
                    # Attempt to reset index to safe state
                    try:
                        self.reset()
                    except Exception:
                        pass
                    success = False

                # Update job statuses
                finished_at = datetime.utcnow().isoformat()
                status_val = "completed" if success else "failed"
                with sqlite3.connect(str(self._db_path), timeout=30) as conn:
                    cur = conn.cursor()
                    cur.executemany(
                        "UPDATE deletion_jobs SET status = ?, enqueued_at = ? WHERE id = ?",
                        ((status_val, finished_at, jid) for jid in job_ids),
                    )
                    conn.commit()
            except Exception:
                # Avoid allowing exceptions to bubble to threadpool
                pass

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
