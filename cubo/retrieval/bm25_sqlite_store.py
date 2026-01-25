"""
SQLite-based BM25 store implementation using FTS5.
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from cubo.config import config
from cubo.retrieval.bm25_store import BM25Store


class BM25SqliteStore(BM25Store):
    """
    BM25 store implementation backed by SQLite FTS5.

    Uses FTS5's built-in bm25() function for scoring.
    Provides scalable full-text search without loading the entire index into RAM.
    """

    def __init__(self, index_dir: Optional[str] = None, **kwargs):
        """Initialize SQLite BM25 store.

        Args:
            index_dir: Directory where bm25.db will be stored.
            **kwargs: Ignored arguments for compatibility.
        """
        self.index_dir = Path(index_dir or config.get("bm25_input_dir", "./data"))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.index_dir / "bm25.db"
        self._lock = threading.Lock()

        # Initialize DB
        self._init_db()

        # Cached list of docs (optional, lazy loaded if accessed)
        self._docs_cache = None

    def _init_db(self):
        """Initialize FTS5 tables."""
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Enable WAL mode for concurrency
                conn.execute("PRAGMA journal_mode=WAL")

                # FTS5 table for full-text search
                # id is stored but unindexed in FTS (we use a separate lookup for id->rowid)
                # content is the text to be indexed
                # metadata is JSON blob
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
                        id UNINDEXED,
                        text,
                        metadata UNINDEXED
                    )
                    """
                )

                # Lookup table for fast id -> rowid mapping
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS doc_lookup (
                        id TEXT PRIMARY KEY,
                        fts_rowid INTEGER
                    )
                    """
                )

                # Index for fast lookup
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lookup_id ON doc_lookup(id)")

    @property
    def docs(self) -> List[Dict]:
        """Return all documents.

        WARNING: This loads all docs from DB into RAM. Use with caution on large datasets.
        Kept for backward compatibility.
        """
        if self._docs_cache is not None:
            return self._docs_cache

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT id, text, metadata FROM docs_fts")
            results = []
            for row in cursor:
                results.append(
                    {
                        "doc_id": row[0],
                        "text": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                    }
                )
        self._docs_cache = results
        return results

    def index_documents(self, docs: List[Dict]) -> None:
        """Build index from scratch (replaces existing)."""
        if not docs:
            return

        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Drop and recreate to ensure clean slate
                conn.execute("DROP TABLE IF EXISTS doc_lookup")
                conn.execute("DROP TABLE IF EXISTS docs_fts")
                self._init_db()  # Recreate tables

                # Batch insert
                data_fts = []
                data_lookup = []

                # We need to manage rowids manually or let SQLite handle them.
                # Simplest is to insert into FTS, get rowid, then insert into lookup.
                # But executemany doesn't return inserted rowids easily for multiple rows.

                # Strategy: Insert one by one? Too slow.
                # Strategy: Insert into FTS, letting it assign rowids (1..N usually).
                # But we need to map id->rowid.

                # Better Strategy: Contentless FTS? No, we need text.
                # External Content FTS? Complex.

                # Optimization:
                # 1. Insert into FTS.
                # 2. 'insert into doc_lookup select id, rowid from docs_fts' (bulk copy).

                batch = []
                for d in docs:
                    doc_id = d.get("doc_id")
                    text = d.get("text", "")
                    meta = json.dumps(d.get("metadata", {}))
                    if doc_id:
                        batch.append((doc_id, text, meta))

                if not batch:
                    return

                # Bulk insert into FTS
                conn.executemany("INSERT INTO docs_fts(id, text, metadata) VALUES (?, ?, ?)", batch)

                # Populate lookup table (fast bulk copy)
                conn.execute("INSERT INTO doc_lookup(id, fts_rowid) SELECT id, rowid FROM docs_fts")
                conn.commit()

        # Invalidate cache
        self._docs_cache = None

    def add_documents(self, docs: List[Dict], reset: bool = False) -> None:
        """Add documents incrementally."""
        if reset:
            self.index_documents(docs)
            return

        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                for d in docs:
                    doc_id = d.get("doc_id")
                    if not doc_id:
                        continue

                    text = d.get("text", "")
                    meta = json.dumps(d.get("metadata", {}))

                    # Check if exists
                    existing = conn.execute(
                        "SELECT fts_rowid FROM doc_lookup WHERE id = ?", (doc_id,)
                    ).fetchone()
                    if existing:
                        # Update
                        rowid = existing[0]
                        conn.execute(
                            "UPDATE docs_fts SET text = ?, metadata = ? WHERE rowid = ?",
                            (text, meta, rowid),
                        )
                    else:
                        # Insert
                        cur = conn.execute(
                            "INSERT INTO docs_fts(id, text, metadata) VALUES (?, ?, ?)",
                            (doc_id, text, meta),
                        )
                        new_rowid = cur.lastrowid
                        conn.execute(
                            "INSERT INTO doc_lookup(id, fts_rowid) VALUES (?, ?)",
                            (doc_id, new_rowid),
                        )
                conn.commit()

        self._docs_cache = None

    def search(
        self,
        query: str,
        top_k: int = 10,
        docs: Optional[List[Dict]] = None,
        doc_ids: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """Search using FTS5 BM25."""
        if not query:
            return []

        # ... (sanitization logic remains the same)
        safe_query = query.replace('"', '""')  # minimal escaping

        # Tokenize simply to construct an OR query
        tokens = ["".join(c for c in w if c.isalnum()) for w in query.split()]
        tokens = [t for t in tokens if t and len(t) <= 64]
        if not tokens:
            return []

        assert all(t.isalnum() for t in tokens), "FTS tokens must be alphanumeric"
        max_tokens = 50
        tokens = tokens[:max_tokens]
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            sql = """
                SELECT id, text, metadata, bm25(docs_fts) as score 
                FROM docs_fts 
                WHERE docs_fts MATCH ? 
                ORDER BY score ASC 
                LIMIT ?
            """

            # Combine docs subset and doc_ids filter
            allowed_ids = doc_ids
            if docs is not None:
                subset_ids = {d.get("doc_id") for d in docs if d.get("doc_id")}
                if allowed_ids:
                    allowed_ids = allowed_ids.intersection(subset_ids)
                else:
                    allowed_ids = subset_ids

            cursor = conn.execute(
                sql, (fts_query, top_k * 10 if allowed_ids else top_k)
            )  # nosec: B608

            for row in cursor:
                doc_id = row[0]
                if allowed_ids and doc_id not in allowed_ids:
                    continue

                score = row[3]  # negative value
                # Convert to positive similarity for compatibility (0..1 or just positive)
                # BM25 is unbound, but usually we want higher = better.
                # similarity = -score
                similarity = -1.0 * score

                results.append(
                    {
                        "doc_id": doc_id,
                        "text": row[1],
                        "metadata": json.loads(row[2]) if row[2] else {},
                        "similarity": similarity,
                    }
                )

                if len(results) >= top_k:
                    break

        return results

    def compute_score(
        self, query_terms: List[str], doc_id: str, doc_text: Optional[str] = None
    ) -> float:
        """Compute BM25 score for a single document."""
        if not query_terms:
            return 0.0

        # Construct query from terms
        fts_query = " OR ".join(f'"{t}"' for t in query_terms)

        with sqlite3.connect(str(self.db_path)) as conn:
            # We need to find the rowid for the doc_id to match specific row

            # If doc_text is provided, we technically should score THAT text.
            # FTS5 can't easily score text not in the table.
            # But the interface says "doc_text: Optional".
            # If doc_text is passed, we might be scoring a candidate not yet indexed?
            # BM25PythonStore handles this.
            # If we strictly use SQLite, we can't score unindexed text without inserting it.
            # Fallback: if doc_id not in DB, return 0.0?
            # Or if doc_text is provided, ignore DB and use Python implementation?
            # That requires `term_doc_freq` etc. which we don't have.
            # So we only support scoring indexed docs.

            row = conn.execute(
                "SELECT fts_rowid FROM doc_lookup WHERE id = ?", (doc_id,)
            ).fetchone()
            if not row:
                return 0.0

            fts_rowid = row[0]

            # Score specific row
            # "SELECT bm25(docs_fts) FROM docs_fts WHERE docs_fts MATCH ? AND rowid = ?"
            sql = "SELECT bm25(docs_fts) FROM docs_fts WHERE docs_fts MATCH ? AND rowid = ?"
            score_row = conn.execute(sql, (fts_query, fts_rowid)).fetchone()

            if score_row:
                return -1.0 * score_row[0]

            return 0.0

    def load_stats(self, path: str):
        """No-op for SQLite store (stats are internal to FTS5)."""
        pass

    def save_stats(self, path: str):
        """No-op for SQLite store."""
        pass

    def close(self):
        """Close connections (handled by context managers usually)."""
        pass
