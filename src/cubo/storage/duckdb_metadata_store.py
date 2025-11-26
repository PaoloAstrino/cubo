"""
DuckDB-based Metadata Store for searchable document metadata.

This module provides a high-performance metadata storage layer using DuckDB,
offering:
- Columnar storage for efficient analytics
- Full-text search on document content
- SQL query interface for flexible metadata queries
- JSON column support for arbitrary metadata
- Integration with FAISS and BM25 indices

The store maintains document metadata separately from vector embeddings,
allowing for fast metadata filtering before or after vector search.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None

from src.cubo.config import config
from src.cubo.utils.logger import logger


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""
    pass


class QueryError(MetadataStoreError):
    """Error executing a query."""
    pass


class SchemaError(MetadataStoreError):
    """Error with table schema."""
    pass


class DocumentStatus(str, Enum):
    """Status of a document in the store."""
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class DocumentMetadata:
    """Metadata for a single document."""
    doc_id: str
    text: str
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    language: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "status": self.status.value if isinstance(self.status, DocumentStatus) else self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "language": self.language,
            "custom_metadata": self.custom_metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = DocumentStatus(status)
        
        def parse_datetime(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)
        
        return cls(
            doc_id=data["doc_id"],
            text=data.get("text", ""),
            source_file=data.get("source_file"),
            chunk_index=data.get("chunk_index"),
            total_chunks=data.get("total_chunks"),
            status=status,
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            indexed_at=parse_datetime(data.get("indexed_at")),
            file_type=data.get("file_type"),
            file_size=data.get("file_size"),
            word_count=data.get("word_count"),
            char_count=data.get("char_count"),
            language=data.get("language"),
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class QueryResult:
    """Result of a metadata query."""
    documents: List[DocumentMetadata]
    total_count: int
    query_time_ms: float
    has_more: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "documents": [d.to_dict() for d in self.documents],
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
            "has_more": self.has_more,
        }


class DuckDBMetadataStore:
    """
    High-performance metadata store using DuckDB.
    
    Provides:
    - Fast columnar storage for document metadata
    - SQL query interface
    - Full-text search support
    - JSON metadata columns
    - Efficient batch operations
    
    Usage:
        store = DuckDBMetadataStore()
        store.initialize()
        
        # Add documents
        store.add_document(DocumentMetadata(doc_id="1", text="Hello world"))
        
        # Query by metadata
        results = store.query(where={"source_file": "doc.pdf"})
        
        # Full-text search
        results = store.search_text("hello")
        
        # SQL queries
        results = store.execute_query("SELECT * FROM documents WHERE word_count > 100")
    """
    
    _instance: Optional["DuckDBMetadataStore"] = None
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
        db_path: Optional[str] = None,
        read_only: bool = False,
        in_memory: bool = False,
    ):
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to database file (default from config)
            read_only: Open in read-only mode
            in_memory: Use in-memory database (for testing)
        """
        if self._initialized:
            return
        
        if not DUCKDB_AVAILABLE:
            logger.warning("DuckDB not available, falling back to SQLite-compatible mode")
            self._use_fallback = True
        else:
            self._use_fallback = False
        
        if in_memory:
            self._db_path = ":memory:"
        else:
            self._db_path = db_path or config.get(
                "metadata_store.duckdb_path", 
                str(Path(config.get("data_dir", "./data")) / "metadata.duckdb")
            )
        
        self._read_only = read_only
        self._in_memory = in_memory
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._local = threading.local()
        
        # Statistics
        self._stats = {
            "documents_added": 0,
            "documents_updated": 0,
            "queries_executed": 0,
            "total_query_time_ms": 0.0,
        }
        
        self._initialized = True
        logger.info(f"DuckDBMetadataStore initialized (path={self._db_path})")
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            if self._use_fallback:
                raise MetadataStoreError("DuckDB not available")
            
            # Ensure directory exists
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self._local.conn = duckdb.connect(
                self._db_path,
                read_only=self._read_only,
            )
        return self._local.conn
    
    @contextmanager
    def _transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    def initialize(self) -> None:
        """Initialize the database schema."""
        if self._use_fallback:
            logger.warning("Using fallback mode - limited functionality")
            return
        
        conn = self._get_connection()
        
        # Create main documents table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id VARCHAR PRIMARY KEY,
                text VARCHAR,
                source_file VARCHAR,
                chunk_index INTEGER,
                total_chunks INTEGER,
                status VARCHAR DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                indexed_at TIMESTAMP,
                file_type VARCHAR,
                file_size BIGINT,
                word_count INTEGER,
                char_count INTEGER,
                language VARCHAR,
                custom_metadata JSON
            )
        """)
        
        # Create full-text search index using DuckDB FTS extension
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            # Check if FTS index exists before creating
            try:
                conn.execute("""
                    PRAGMA create_fts_index('documents', 'doc_id', 'text', overwrite=1)
                """)
            except Exception:
                # Index may already exist or FTS not fully supported
                pass
        except Exception as e:
            logger.warning(f"Full-text search extension not available: {e}")
        
        # Create indices for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_source_file 
            ON documents(source_file)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_status 
            ON documents(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_created_at 
            ON documents(created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_file_type 
            ON documents(file_type)
        """)
        
        # Create ingestion runs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                run_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_folder VARCHAR,
                chunks_count INTEGER,
                output_parquet VARCHAR,
                status VARCHAR DEFAULT 'pending',
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                error_message VARCHAR,
                metadata JSON
            )
        """)
        
        # Create chunk mappings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_mappings (
                run_id VARCHAR,
                old_id VARCHAR,
                new_id VARCHAR,
                mapping_metadata JSON,
                PRIMARY KEY (run_id, old_id)
            )
        """)
        
        # Create index versions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_versions (
                version_id VARCHAR PRIMARY KEY,
                index_dir VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                document_count INTEGER,
                index_type VARCHAR,
                metadata JSON
            )
        """)
        
        logger.info("DuckDB schema initialized")
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    def add_document(self, doc: DocumentMetadata) -> None:
        """
        Add a single document to the store.
        
        Args:
            doc: Document metadata to add
        """
        self.add_documents([doc])
    
    def add_documents(self, docs: List[DocumentMetadata]) -> int:
        """
        Add multiple documents to the store.
        
        Args:
            docs: List of document metadata to add
            
        Returns:
            Number of documents added
        """
        if not docs:
            return 0
        
        if self._use_fallback:
            raise MetadataStoreError("DuckDB not available")
        
        with self._transaction() as conn:
            now = datetime.utcnow()
            
            for doc in docs:
                # Auto-compute word/char count if not provided
                if doc.word_count is None and doc.text:
                    doc.word_count = len(doc.text.split())
                if doc.char_count is None and doc.text:
                    doc.char_count = len(doc.text)
                if doc.created_at is None:
                    doc.created_at = now
                
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        doc_id, text, source_file, chunk_index, total_chunks,
                        status, created_at, updated_at, indexed_at,
                        file_type, file_size, word_count, char_count,
                        language, custom_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    doc.doc_id,
                    doc.text,
                    doc.source_file,
                    doc.chunk_index,
                    doc.total_chunks,
                    doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                    doc.created_at,
                    doc.updated_at,
                    doc.indexed_at,
                    doc.file_type,
                    doc.file_size,
                    doc.word_count,
                    doc.char_count,
                    doc.language,
                    json.dumps(doc.custom_metadata) if doc.custom_metadata else None,
                ])
            
            self._stats["documents_added"] += len(docs)
            
        logger.debug(f"Added {len(docs)} documents to metadata store")
        return len(docs)
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentMetadata if found, None otherwise
        """
        results = self.get_documents([doc_id])
        return results[0] if results else None
    
    def get_documents(self, doc_ids: List[str]) -> List[DocumentMetadata]:
        """
        Get multiple documents by ID.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of DocumentMetadata (in same order as input, missing docs omitted)
        """
        if not doc_ids or self._use_fallback:
            return []
        
        with self._transaction() as conn:
            placeholders = ", ".join(["?" for _ in doc_ids])
            result = conn.execute(f"""
                SELECT doc_id, text, source_file, chunk_index, total_chunks,
                       status, created_at, updated_at, indexed_at,
                       file_type, file_size, word_count, char_count,
                       language, custom_metadata
                FROM documents
                WHERE doc_id IN ({placeholders})
            """, doc_ids).fetchall()
            
            return [self._row_to_document(row) for row in result]
    
    def update_document(
        self, 
        doc_id: str, 
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update a document's metadata.
        
        Args:
            doc_id: Document ID
            updates: Dictionary of fields to update
            
        Returns:
            True if document was updated
        """
        if not updates or self._use_fallback:
            return False
        
        # Filter to allowed fields
        allowed_fields = {
            "text", "source_file", "chunk_index", "total_chunks",
            "status", "indexed_at", "file_type", "file_size",
            "word_count", "char_count", "language", "custom_metadata",
        }
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        # Always update updated_at
        updates["updated_at"] = datetime.utcnow()
        
        with self._transaction() as conn:
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values())
            
            # Handle JSON field
            for i, (k, v) in enumerate(updates.items()):
                if k == "custom_metadata" and isinstance(v, dict):
                    values[i] = json.dumps(v)
                if k == "status" and isinstance(v, DocumentStatus):
                    values[i] = v.value
            
            values.append(doc_id)
            
            result = conn.execute(f"""
                UPDATE documents SET {set_clause} WHERE doc_id = ?
            """, values)
            
            self._stats["documents_updated"] += 1
            
            return True
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the store.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted
        """
        return self.delete_documents([doc_id]) > 0
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """
        Delete multiple documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Number of documents deleted
        """
        if not doc_ids or self._use_fallback:
            return 0
        
        with self._transaction() as conn:
            placeholders = ", ".join(["?" for _ in doc_ids])
            conn.execute(f"""
                DELETE FROM documents WHERE doc_id IN ({placeholders})
            """, doc_ids)
            
            return len(doc_ids)
    
    def mark_indexed(self, doc_ids: List[str]) -> int:
        """
        Mark documents as indexed.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Number of documents updated
        """
        if not doc_ids or self._use_fallback:
            return 0
        
        with self._transaction() as conn:
            now = datetime.utcnow()
            placeholders = ", ".join(["?" for _ in doc_ids])
            conn.execute(f"""
                UPDATE documents 
                SET status = 'indexed', indexed_at = ?, updated_at = ?
                WHERE doc_id IN ({placeholders})
            """, [now, now] + doc_ids)
            
            return len(doc_ids)
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def query(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_text: bool = True,
    ) -> QueryResult:
        """
        Query documents by metadata filters.
        
        Args:
            where: Filter conditions as field: value pairs
            order_by: Column to order by (prefix with - for DESC)
            limit: Maximum results to return
            offset: Number of results to skip
            include_text: Whether to include document text
            
        Returns:
            QueryResult with matching documents
        """
        import time
        start_time = time.time()
        
        if self._use_fallback:
            return QueryResult(documents=[], total_count=0, query_time_ms=0.0)
        
        # Build query - always use same column order: doc_id, text, source_file, ...
        if include_text:
            select_cols = """
                doc_id, text, source_file, chunk_index, total_chunks,
                status, created_at, updated_at, indexed_at,
                file_type, file_size, word_count, char_count,
                language, custom_metadata
            """
        else:
            select_cols = """
                doc_id, '' as text, source_file, chunk_index, total_chunks,
                status, created_at, updated_at, indexed_at,
                file_type, file_size, word_count, char_count,
                language, custom_metadata
            """
        
        conditions = []
        params = []
        
        if where:
            for key, value in where.items():
                if value is None:
                    conditions.append(f"{key} IS NULL")
                elif isinstance(value, list):
                    placeholders = ", ".join(["?" for _ in value])
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                elif isinstance(value, dict):
                    # Support operators like {"$gt": 100, "$lt": 200}
                    for op, val in value.items():
                        if op == "$gt":
                            conditions.append(f"{key} > ?")
                            params.append(val)
                        elif op == "$gte":
                            conditions.append(f"{key} >= ?")
                            params.append(val)
                        elif op == "$lt":
                            conditions.append(f"{key} < ?")
                            params.append(val)
                        elif op == "$lte":
                            conditions.append(f"{key} <= ?")
                            params.append(val)
                        elif op == "$ne":
                            conditions.append(f"{key} != ?")
                            params.append(val)
                        elif op == "$like":
                            conditions.append(f"{key} LIKE ?")
                            params.append(val)
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Handle ordering
        order_clause = "created_at DESC"
        if order_by:
            if order_by.startswith("-"):
                order_clause = f"{order_by[1:]} DESC"
            else:
                order_clause = f"{order_by} ASC"
        
        with self._transaction() as conn:
            # Get total count
            count_result = conn.execute(f"""
                SELECT COUNT(*) FROM documents WHERE {where_clause}
            """, params).fetchone()
            total_count = count_result[0] if count_result else 0
            
            # Get documents
            result = conn.execute(f"""
                SELECT {select_cols}
                FROM documents
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
            """, params + [limit, offset]).fetchall()
            
            documents = [self._row_to_document(row) for row in result]
        
        query_time = (time.time() - start_time) * 1000
        self._stats["queries_executed"] += 1
        self._stats["total_query_time_ms"] += query_time
        
        return QueryResult(
            documents=documents,
            total_count=total_count,
            query_time_ms=query_time,
            has_more=(offset + len(documents)) < total_count,
        )
    
    def search_text(
        self,
        query: str,
        limit: int = 100,
        include_text: bool = True,
    ) -> QueryResult:
        """
        Full-text search on document content.
        
        Args:
            query: Search query string
            limit: Maximum results
            include_text: Whether to include full text
            
        Returns:
            QueryResult with matching documents
        """
        import time
        start_time = time.time()
        
        if self._use_fallback or not query:
            return QueryResult(documents=[], total_count=0, query_time_ms=0.0)
        
        with self._transaction() as conn:
            # Use LIKE as fallback if FTS not available
            try:
                result = conn.execute("""
                    SELECT doc_id, text, source_file, chunk_index, total_chunks,
                           status, created_at, updated_at, indexed_at,
                           file_type, file_size, word_count, char_count,
                           language, custom_metadata
                    FROM documents
                    WHERE text ILIKE ?
                    LIMIT ?
                """, [f"%{query}%", limit]).fetchall()
            except Exception:
                result = []
            
            documents = [self._row_to_document(row) for row in result]
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            documents=documents,
            total_count=len(documents),
            query_time_ms=query_time,
            has_more=False,
        )
    
    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.
        
        Args:
            sql: SQL query string
            params: Optional query parameters
            
        Returns:
            List of result rows as dictionaries
        """
        if self._use_fallback:
            return []
        
        with self._transaction() as conn:
            result = conn.execute(sql, params or [])
            columns = [desc[0] for desc in result.description or []]
            rows = result.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    # =========================================================================
    # Ingestion Run Operations
    # =========================================================================
    
    def record_ingestion_run(
        self,
        run_id: str,
        source_folder: str,
        chunks_count: int,
        output_parquet: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a new ingestion run."""
        if self._use_fallback:
            return
        
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_runs 
                (run_id, source_folder, chunks_count, output_parquet, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                run_id,
                source_folder,
                chunks_count,
                output_parquet,
                json.dumps(metadata) if metadata else None,
            ])
    
    def update_ingestion_status(
        self,
        run_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update ingestion run status."""
        if self._use_fallback:
            return
        
        with self._transaction() as conn:
            conn.execute("""
                UPDATE ingestion_runs 
                SET status = ?, started_at = ?, finished_at = ?, error_message = ?
                WHERE run_id = ?
            """, [status, started_at, finished_at, error_message, run_id])
    
    def get_ingestion_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get ingestion run by ID."""
        results = self.execute_query(
            "SELECT * FROM ingestion_runs WHERE run_id = ?",
            [run_id]
        )
        return results[0] if results else None
    
    # =========================================================================
    # Index Version Operations
    # =========================================================================
    
    def record_index_version(
        self,
        version_id: str,
        index_dir: str,
        document_count: int = 0,
        index_type: str = "faiss",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a new index version."""
        if self._use_fallback:
            return
        
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO index_versions 
                (version_id, index_dir, document_count, index_type, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, [
                version_id,
                index_dir,
                document_count,
                index_type,
                json.dumps(metadata) if metadata else None,
            ])
    
    def get_latest_index_version(self, index_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the latest index version."""
        if index_type:
            results = self.execute_query(
                "SELECT * FROM index_versions WHERE index_type = ? ORDER BY created_at DESC LIMIT 1",
                [index_type]
            )
        else:
            results = self.execute_query(
                "SELECT * FROM index_versions ORDER BY created_at DESC LIMIT 1"
            )
        return results[0] if results else None
    
    def list_index_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent index versions."""
        return self.execute_query(
            "SELECT * FROM index_versions ORDER BY created_at DESC LIMIT ?",
            [limit]
        )
    
    # =========================================================================
    # Statistics and Maintenance
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        stats = dict(self._stats)
        
        if not self._use_fallback:
            with self._transaction() as conn:
                # Document counts by status
                result = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM documents
                    GROUP BY status
                """).fetchall()
                stats["documents_by_status"] = {row[0]: row[1] for row in result}
                
                # Total documents
                result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                stats["total_documents"] = result[0] if result else 0
                
                # Average word count
                result = conn.execute(
                    "SELECT AVG(word_count) FROM documents WHERE word_count IS NOT NULL"
                ).fetchone()
                stats["avg_word_count"] = round(result[0], 2) if result and result[0] else 0
                
                # File type distribution
                result = conn.execute("""
                    SELECT file_type, COUNT(*) as count
                    FROM documents
                    WHERE file_type IS NOT NULL
                    GROUP BY file_type
                """).fetchall()
                stats["documents_by_file_type"] = {row[0]: row[1] for row in result}
        
        return stats
    
    def vacuum(self) -> None:
        """Optimize the database by running VACUUM."""
        if self._use_fallback:
            return
        
        conn = self._get_connection()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")
    
    def _row_to_document(self, row: tuple) -> DocumentMetadata:
        """Convert a database row to DocumentMetadata."""
        # Row order from query: doc_id, text, source_file, chunk_index, total_chunks,
        #                       status, created_at, updated_at, indexed_at,
        #                       file_type, file_size, word_count, char_count,
        #                       language, custom_metadata
        doc_id, text, source_file, chunk_index, total_chunks, status, \
        created_at, updated_at, indexed_at, file_type, file_size, \
        word_count, char_count, language, custom_metadata = row
        
        return DocumentMetadata(
            doc_id=doc_id,
            text=text or "",
            source_file=source_file,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            status=DocumentStatus(status) if status else DocumentStatus.PENDING,
            created_at=created_at,
            updated_at=updated_at,
            indexed_at=indexed_at,
            file_type=file_type,
            file_size=file_size,
            word_count=word_count,
            char_count=char_count,
            language=language,
            custom_metadata=json.loads(custom_metadata) if custom_metadata else {},
        )


# ============================================================================
# Singleton accessor
# ============================================================================

_metadata_store: Optional[DuckDBMetadataStore] = None


def get_duckdb_metadata_store(
    db_path: Optional[str] = None,
    in_memory: bool = False,
) -> DuckDBMetadataStore:
    """
    Get the global DuckDB metadata store instance.
    
    Args:
        db_path: Optional custom database path
        in_memory: Use in-memory database
        
    Returns:
        DuckDBMetadataStore instance
    """
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = DuckDBMetadataStore(db_path=db_path, in_memory=in_memory)
        _metadata_store.initialize()
    return _metadata_store
