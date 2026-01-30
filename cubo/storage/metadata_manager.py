"""Simple SQLite metadata manager for ingestion runs, chunk mappings, and index versions.
This module provides a lightweight SQLite wrapper; no external dependencies beyond stdlib.
"""

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from cubo.config import config
from cubo.utils.logger import logger


class MetadataManager:
    def __init__(self, db_path: Optional[str] = None):
        # Default to ./storage/metadata.db if not provided or configured differently
        # This keeps the 'data' folder clean for user documents only.
        default_path = config.get("metadata_db_path", "./storage/metadata.db")
        self.db_path = Path(db_path or default_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Allow cross-thread use with a simple lock guard; set a small timeout to avoid busy errors.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=5.0)

        # Enable WAL mode for better concurrency (readers don't block writers)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except Exception as e:
            logger.warning(f"Failed to enable WAL mode for metadata DB: {e}")

        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                source_folder TEXT,
                chunks_count INTEGER,
                output_parquet TEXT,
                status TEXT,
                started_at TEXT,
                finished_at TEXT
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_files (
                run_id TEXT,
                file_path TEXT,
                status TEXT,
                error TEXT,
                attempts INTEGER DEFAULT 0,
                size_bytes INTEGER,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (run_id, file_path)
            )
        """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_files_run_id ON ingestion_files (run_id)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_mappings (
                run_id TEXT,
                old_id TEXT,
                new_id TEXT,
                metadata TEXT,
                primary key (run_id, old_id)
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS index_versions (
                id TEXT PRIMARY KEY,
                index_dir TEXT,
                created_at TEXT
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scaffold_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                scaffold_dir TEXT,
                model_version TEXT,
                scaffold_count INTEGER,
                manifest_path TEXT
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scaffold_mappings (
                run_id TEXT,
                scaffold_id TEXT,
                chunk_id TEXT,
                metadata TEXT,
                PRIMARY KEY (run_id, scaffold_id, chunk_id)
            )
        """
        )
        self.conn.commit()
        # Ensure migration compatibility: add missing columns if needed
        try:
            cur.execute("PRAGMA table_info(ingestion_runs)")
            cols = set(r[1] for r in cur.fetchall())
            if "status" not in cols:
                cur.execute("ALTER TABLE ingestion_runs ADD COLUMN status TEXT")
            if "started_at" not in cols:
                cur.execute("ALTER TABLE ingestion_runs ADD COLUMN started_at TEXT")
            if "finished_at" not in cols:
                cur.execute("ALTER TABLE ingestion_runs ADD COLUMN finished_at TEXT")
            self.conn.commit()
        except Exception:
            # If this fails for any reason, we ignore to keep compatibility
            pass
        # Ensure ingestion_files has expected columns
        try:
            cur.execute("PRAGMA table_info(ingestion_files)")
            cols = set(r[1] for r in cur.fetchall())
            if "attempts" not in cols:
                cur.execute("ALTER TABLE ingestion_files ADD COLUMN attempts INTEGER DEFAULT 0")
            if "size_bytes" not in cols:
                cur.execute("ALTER TABLE ingestion_files ADD COLUMN size_bytes INTEGER")
            if "created_at" not in cols:
                cur.execute("ALTER TABLE ingestion_files ADD COLUMN created_at TEXT")
            if "updated_at" not in cols:
                cur.execute("ALTER TABLE ingestion_files ADD COLUMN updated_at TEXT")
            self.conn.commit()
        except Exception:
            pass
        # Ensure scaffold_runs has expected columns (for older DBs/migrations)
        try:
            cur.execute("PRAGMA table_info(scaffold_runs)")
            cols = set(r[1] for r in cur.fetchall())
            if "model_version" not in cols:
                cur.execute("ALTER TABLE scaffold_runs ADD COLUMN model_version TEXT")
            if "scaffold_count" not in cols:
                cur.execute("ALTER TABLE scaffold_runs ADD COLUMN scaffold_count INTEGER")
            if "manifest_path" not in cols:
                cur.execute("ALTER TABLE scaffold_runs ADD COLUMN manifest_path TEXT")
            self.conn.commit()
        except Exception:
            # Ignore failures; older DBs may not have scaffold_runs table and we'll rely on CREATE TABLE IF NOT EXISTS
            pass

        # --- Chat Persistence Schema ---
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                title TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT,
                metadata TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id)"
        )
        self.conn.commit()

        # Ensure scaffold_mappings has expected columns (for older DBs/migrations)
        try:
            cur.execute("PRAGMA table_info(scaffold_mappings)")
            cols = set(r[1] for r in cur.fetchall())
            # If metadata is missing ensure to add it to allow storing JSON metadata
            if "metadata" not in cols:
                cur.execute("ALTER TABLE scaffold_mappings ADD COLUMN metadata TEXT")
            # If run_id is missing, try to migrate by recreating the table with run_id
            if "run_id" not in cols:
                logger.info(
                    "scaffold_mappings missing run_id - attempting migration to add run_id column"
                )
                try:
                    # Create a new table with run_id column
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS scaffold_mappings_new (
                        run_id TEXT,
                        scaffold_id TEXT,
                        chunk_id TEXT,
                        metadata TEXT,
                        PRIMARY KEY (run_id, scaffold_id, chunk_id)
                    )"""
                    )
                    # Copy existing rows into new table with empty run_id
                    cur.execute(
                        """INSERT OR REPLACE INTO scaffold_mappings_new (run_id, scaffold_id, chunk_id, metadata) SELECT '', scaffold_id, chunk_id, metadata FROM scaffold_mappings"""
                    )
                    # Rename old table and new table swap
                    cur.execute("ALTER TABLE scaffold_mappings RENAME TO scaffold_mappings_old")
                    cur.execute("ALTER TABLE scaffold_mappings_new RENAME TO scaffold_mappings")
                    cur.execute("DROP TABLE IF EXISTS scaffold_mappings_old")
                    logger.info(
                        "scaffold_mappings migration completed: run_id added with empty default for existing rows"
                    )
                except Exception as e:
                    # If migration fails, log a warning and continue
                    logger.warning(f"scaffold_mappings migration to add run_id failed: {e}")
            self.conn.commit()
        except Exception:
            pass

    def record_ingestion_run(
        self,
        run_id: str,
        source_folder: str,
        chunks_count: int,
        output_parquet: Optional[str] = None,
        status: str = "pending",
    ) -> None:
        with self._lock:
            cur = self.conn.cursor()
            # Default status: pending (fast pass not yet completed)
            cur.execute(
                """INSERT OR REPLACE INTO ingestion_runs (id, created_at, source_folder, chunks_count, output_parquet, status) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    datetime.datetime.utcnow().isoformat(),
                    source_folder,
                    chunks_count,
                    output_parquet,
                    status,
                ),
            )
            self.conn.commit()

    # --- Ingestion file status helpers ---
    def _ensure_file_row(
        self,
        run_id: str,
        file_path: str,
        size_bytes: Optional[int] = None,
        default_status: str = "queued",
    ) -> None:
        now = datetime.datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO ingestion_files (run_id, file_path, status, error, attempts, size_bytes, created_at, updated_at)
                VALUES (?, ?, ?, NULL, 0, ?, ?, ?)
                """,
                (run_id, file_path, default_status, size_bytes, now, now),
            )
            self.conn.commit()

    def set_file_status(
        self,
        run_id: str,
        file_path: str,
        status: str,
        error: Optional[str] = None,
        size_bytes: Optional[int] = None,
        increment_attempt: bool = False,
    ) -> None:
        """Upsert or update a file row with status, optionally increment attempts."""
        self._ensure_file_row(run_id, file_path, size_bytes=size_bytes, default_status=status)
        now = datetime.datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                UPDATE ingestion_files
                SET status = ?,
                    error = ?,
                    size_bytes = COALESCE(?, size_bytes),
                    attempts = attempts + ?,
                    updated_at = ?
                WHERE run_id = ? AND file_path = ?
                """,
                (status, error, size_bytes, 1 if increment_attempt else 0, now, run_id, file_path),
            )
            self.conn.commit()

    def mark_file_processing(
        self, run_id: str, file_path: str, size_bytes: Optional[int] = None
    ) -> None:
        self.set_file_status(run_id, file_path, "processing", size_bytes=size_bytes)

    def mark_file_succeeded(
        self, run_id: str, file_path: str, size_bytes: Optional[int] = None
    ) -> None:
        self.set_file_status(run_id, file_path, "succeeded", size_bytes=size_bytes)

    def mark_file_failed(
        self, run_id: str, file_path: str, error: str, size_bytes: Optional[int] = None
    ) -> None:
        self.set_file_status(
            run_id, file_path, "failed", error=error, size_bytes=size_bytes, increment_attempt=True
        )

    def list_files_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT file_path, status, error, attempts, size_bytes, created_at, updated_at
                FROM ingestion_files WHERE run_id = ?
                ORDER BY file_path
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "file_path": r[0],
                "status": r[1],
                "error": r[2],
                "attempts": r[3],
                "size_bytes": r[4],
                "created_at": r[5],
                "updated_at": r[6],
            }
            for r in rows
        ]

    def get_file_status(self, run_id: str, file_path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT file_path, status, error, attempts, size_bytes, created_at, updated_at
                FROM ingestion_files WHERE run_id = ? AND file_path = ?
                """,
                (run_id, file_path),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "file_path": row[0],
            "status": row[1],
            "error": row[2],
            "attempts": row[3],
            "size_bytes": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }

    def get_file_status_counts(self, run_id: str) -> Dict[str, int]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT status, COUNT(*) FROM ingestion_files WHERE run_id = ? GROUP BY status
                """,
                (run_id,),
            )
            rows = cur.fetchall()
        return {r[0]: r[1] for r in rows}

    def update_ingestion_status(
        self,
        run_id: str,
        status: str,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> None:
        with self._lock:
            cur = self.conn.cursor()
            if started_at is None and finished_at is None:
                cur.execute(
                    """UPDATE ingestion_runs SET status = ? WHERE id = ?""", (status, run_id)
                )
            else:
                cur.execute(
                    """UPDATE ingestion_runs SET status = ?, started_at = COALESCE(?, started_at), finished_at = COALESCE(?, finished_at) WHERE id = ?""",
                    (status, started_at, finished_at, run_id),
                )
            self.conn.commit()

    def update_ingestion_run_details(
        self,
        run_id: str,
        chunks_count: Optional[int] = None,
        output_parquet: Optional[str] = None,
        status: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> None:
        with self._lock:
            cur = self.conn.cursor()
            fields = []
            values = []
            if chunks_count is not None:
                fields.append("chunks_count = ?")
                values.append(chunks_count)
            if output_parquet is not None:
                fields.append("output_parquet = ?")
                values.append(output_parquet)
            if status is not None:
                fields.append("status = ?")
                values.append(status)
            if finished_at is not None:
                fields.append("finished_at = ?")
                values.append(finished_at)

            if not fields:
                return

            values.append(run_id)
            # Whitelist columns to avoid SQL injection via dynamic field names
            allowed = {"chunks_count", "output_parquet", "status", "finished_at"}
            # Each item in `fields` looks like "<column> = ?" - ensure column is allowed
            for f in fields:
                col = f.split("=")[0].strip()
                if col not in allowed:
                    raise ValueError(f"Unexpected field in update: {col}")

            set_clause = ", ".join(fields)
            query = "UPDATE ingestion_runs SET " + set_clause + " WHERE id = ?"  # nosec B608
            cur.execute(query, tuple(values))
            self.conn.commit()

    def add_chunk_mapping(
        self, run_id: str, old_id: str, new_id: str, metadata: Dict[str, Any]
    ) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO chunk_mappings (run_id, old_id, new_id, metadata) VALUES (?, ?, ?, ?)""",
                (run_id, old_id, new_id, json.dumps(metadata)),
            )
            self.conn.commit()

    def record_scaffold_run(
        self,
        run_id: str,
        scaffold_dir: str,
        model_version: str,
        scaffold_count: int,
        manifest_path: str,
    ) -> None:
        # Use short-lived sqlite connection to avoid Windows cross-thread issues
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO scaffold_runs (id, created_at, scaffold_dir, model_version, scaffold_count, manifest_path) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    datetime.datetime.utcnow().isoformat(),
                    scaffold_dir,
                    model_version,
                    scaffold_count,
                    manifest_path,
                ),
            )
            conn.commit()

    def add_scaffold_mapping(
        self, run_id: str, scaffold_id: str, chunk_id: str, metadata: Dict[str, Any]
    ) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO scaffold_mappings (run_id, scaffold_id, chunk_id, metadata) VALUES (?, ?, ?, ?)""",
                (run_id, scaffold_id, chunk_id, json.dumps(metadata)),
            )
            conn.commit()

    def get_latest_scaffold_run(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT id, scaffold_dir, model_version, scaffold_count, manifest_path, created_at FROM scaffold_runs ORDER BY created_at DESC LIMIT 1"""
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "scaffold_dir": row[1],
            "model_version": row[2],
            "scaffold_count": row[3],
            "manifest_path": row[4],
            "created_at": row[5],
        }

    def list_scaffold_mappings_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT scaffold_id, chunk_id, metadata FROM scaffold_mappings WHERE run_id = ?""",
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {"scaffold_id": r[0], "chunk_id": r[1], "metadata": json.loads(r[2]) if r[2] else {}}
            for r in rows
        ]

    def list_mappings_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT old_id, new_id, metadata FROM chunk_mappings WHERE run_id = ?""",
                (run_id,),
            )
            rows = cur.fetchall()
        return [
            {"old_id": r[0], "new_id": r[1], "metadata": json.loads(r[2]) if r[2] else {}}
            for r in rows
        ]

    def record_index_version(self, version_id: str, index_dir: str) -> None:
        # Use a short-lived connection for this write to avoid cross-thread 'check_same_thread' issues
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO index_versions (id, index_dir, created_at) VALUES (?, ?, ?)""",
                (version_id, index_dir, datetime.datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_latest_index_version(self) -> Optional[Dict[str, Any]]:
        # Use a short-lived connection here to avoid cross-thread 'check_same_thread' errors
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, index_dir, created_at FROM index_versions ORDER BY created_at DESC LIMIT 1"""
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "index_dir": row[1], "created_at": row[2]}

    def list_index_versions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        # Use a short-lived connection for list operations (safer for threaded tests on Windows)
        with sqlite3.connect(str(self.db_path)) as conn:
            cur = conn.cursor()
            sql = "SELECT id, index_dir, created_at FROM index_versions ORDER BY created_at DESC"
            if limit is not None:
                cur.execute(sql + " LIMIT ?", (limit,))
            else:
                cur.execute(sql)
            rows = cur.fetchall()
        return [{"id": r[0], "index_dir": r[1], "created_at": r[2]} for r in rows]

    def get_ingestion_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT id, created_at, source_folder, chunks_count, output_parquet, status, started_at, finished_at FROM ingestion_runs WHERE id = ?""",
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "source_folder": row[2],
            "chunks_count": row[3],
            "output_parquet": row[4],
            "status": row[5],
            "started_at": row[6],
            "finished_at": row[7],
        }

    def list_runs_by_status(self, status: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("""SELECT id FROM ingestion_runs WHERE status = ?""", (status,))
            rows = cur.fetchall()
            ids = [r[0] for r in rows]
        return [self.get_ingestion_run(run_id) for run_id in ids]

    # --- Chat Persistence Methods ---

    def create_conversation(self, title: str = "New Chat") -> str:
        """Create a new conversation and return its ID."""
        import uuid

        conv_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO conversations (id, created_at, updated_at, title) VALUES (?, ?, ?, ?)""",
                (conv_id, now, now, title),
            )
            self.conn.commit()
        return conv_id

    def add_message(
        self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Add a message to a conversation."""
        import uuid

        msg_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        with self._lock:
            cur = self.conn.cursor()
            # Verify conversation exists first
            cur.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if not cur.fetchone():
                # Auto-create if not exists (resilience) or raise error.
                # Ideally we create it, but let's stick to explicit creation or error.
                # For robustness, we'll try to insert the conversation if missing, or error.
                # Let's assume it must exist for now, or the caller handles it.
                # Actually, raising an error is safer to detect logic bugs.
                raise ValueError(f"Conversation {conversation_id} does not exist")

            cur.execute(
                """INSERT INTO messages (id, conversation_id, role, content, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (msg_id, conversation_id, role, content, now, metadata_json),
            )
            # Update conversation updated_at
            cur.execute(
                """UPDATE conversations SET updated_at = ? WHERE id = ?""", (now, conversation_id)
            )
            self.conn.commit()
        return msg_id

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation, ordered by creation time."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT id, role, content, created_at, metadata FROM messages
                   WHERE conversation_id = ? ORDER BY created_at ASC""",
                (conversation_id,),
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "role": r[1],
                "content": r[2],
                "created_at": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
            }
            for r in rows
        ]

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent conversations."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT id, title, created_at, updated_at FROM conversations
                   ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            )
            rows = cur.fetchall()

        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific conversation."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?""",
                (conversation_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        return {"id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3]}

    def get_filenames_in_collection(self, collection_id: str) -> List[str]:
        """Get all document filenames associated with a collection.

        Args:
            collection_id: The collection ID to query

        Returns:
            List of document filenames in the collection, or empty list if collection not found
        """
        try:
            with self._lock:
                cur = self.conn.cursor()
                # Query ingestion_files table for files in this collection
                # We assume collection_id is stored in run_id or related metadata
                cur.execute(
                    """
                    SELECT DISTINCT file_path FROM ingestion_files
                    WHERE run_id = ? AND status = 'success'
                """,
                    (collection_id,),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows] if rows else []
        except Exception as e:
            logger.warning(f"Failed to get filenames for collection {collection_id}: {e}")
            return []

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation messages but keep the conversation record.

        Returns True if any messages were removed.
        """
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            rows_deleted = cur.rowcount
            self.conn.commit()
            return rows_deleted > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            rows_deleted = cur.rowcount
            self.conn.commit()
            return rows_deleted > 0


# Expose a module-level manager instance for simple use
_manager: Optional[MetadataManager] = None


def get_metadata_manager() -> MetadataManager:
    global _manager
    if _manager is None:
        _manager = MetadataManager()
    return _manager
