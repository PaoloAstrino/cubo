"""Simple SQLite metadata manager for ingestion runs, chunk mappings, and index versions.
This module provides a lightweight SQLite wrapper; no external dependencies beyond stdlib.
"""
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import datetime

from src.cubo.config import config
from src.cubo.utils.logger import logger


class MetadataManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or config.get('metadata_db_path', './data/metadata.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                source_folder TEXT,
                chunks_count INTEGER,
                output_parquet TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chunk_mappings (
                run_id TEXT,
                old_id TEXT,
                new_id TEXT,
                metadata TEXT,
                primary key (run_id, old_id)
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS index_versions (
                id TEXT PRIMARY KEY,
                index_dir TEXT,
                created_at TEXT
            )
        ''')
        self.conn.commit()

    def record_ingestion_run(self, run_id: str, source_folder: str, chunks_count: int, output_parquet: Optional[str] = None) -> None:
        cur = self.conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO ingestion_runs (id, created_at, source_folder, chunks_count, output_parquet) VALUES (?, ?, ?, ?, ?)''',
                    (run_id, datetime.datetime.utcnow().isoformat(), source_folder, chunks_count, output_parquet))
        self.conn.commit()

    def add_chunk_mapping(self, run_id: str, old_id: str, new_id: str, metadata: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO chunk_mappings (run_id, old_id, new_id, metadata) VALUES (?, ?, ?, ?)''',
                    (run_id, old_id, new_id, json.dumps(metadata)))
        self.conn.commit()

    def list_mappings_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute('''SELECT old_id, new_id, metadata FROM chunk_mappings WHERE run_id = ?''', (run_id,))
        rows = cur.fetchall()
        return [{'old_id': r[0], 'new_id': r[1], 'metadata': json.loads(r[2]) if r[2] else {}} for r in rows]

    def record_index_version(self, version_id: str, index_dir: str) -> None:
        cur = self.conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO index_versions (id, index_dir, created_at) VALUES (?, ?, ?)''',
                    (version_id, index_dir, datetime.datetime.utcnow().isoformat()))
        self.conn.commit()

    def get_latest_index_version(self) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute('''SELECT id, index_dir, created_at FROM index_versions ORDER BY created_at DESC LIMIT 1''')
        row = cur.fetchone()
        if not row:
            return None
        return {'id': row[0], 'index_dir': row[1], 'created_at': row[2]}


# Expose a module-level manager instance for simple use
_manager: Optional[MetadataManager] = None


def get_metadata_manager() -> MetadataManager:
    global _manager
    if _manager is None:
        _manager = MetadataManager()
    return _manager
