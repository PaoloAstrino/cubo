"""
Tests for the DuckDB Metadata Store module.

Tests document metadata storage, querying, and full-text search capabilities.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.cubo.storage.duckdb_metadata_store import (
    DocumentMetadata,
    DocumentStatus,
    DuckDBMetadataStore,
    QueryResult,
    get_duckdb_metadata_store,
)


# Fixture to reset singleton and provide fresh in-memory store
@pytest.fixture
def reset_singleton():
    """Reset DuckDBMetadataStore singleton before each test."""
    DuckDBMetadataStore._instance = None
    import src.cubo.storage.duckdb_metadata_store as store_module

    store_module._metadata_store = None
    yield
    # Cleanup
    DuckDBMetadataStore._instance = None
    store_module._metadata_store = None


@pytest.fixture
def store(reset_singleton):
    """Provide a fresh in-memory store for each test."""
    s = DuckDBMetadataStore(in_memory=True)
    s.initialize()
    return s


# ============================================================================
# DocumentMetadata Tests
# ============================================================================


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_default_values(self):
        """Test default DocumentMetadata values."""
        doc = DocumentMetadata(doc_id="doc1", text="Hello world")
        assert doc.doc_id == "doc1"
        assert doc.text == "Hello world"
        assert doc.status == DocumentStatus.PENDING
        assert doc.custom_metadata == {}
        assert doc.word_count is None
        assert doc.source_file is None

    def test_with_all_fields(self):
        """Test DocumentMetadata with all fields populated."""
        now = datetime.now()
        doc = DocumentMetadata(
            doc_id="doc2",
            text="Test document content",
            source_file="test.pdf",
            chunk_index=0,
            total_chunks=5,
            status=DocumentStatus.INDEXED,
            created_at=now,
            updated_at=now,
            indexed_at=now,
            file_type="pdf",
            file_size=1024,
            word_count=3,
            char_count=21,
            language="en",
            custom_metadata={"author": "Test"},
        )
        assert doc.source_file == "test.pdf"
        assert doc.status == DocumentStatus.INDEXED
        assert doc.file_type == "pdf"
        assert doc.custom_metadata["author"] == "Test"

    def test_to_dict(self):
        """Test DocumentMetadata serialization."""
        now = datetime.now()
        doc = DocumentMetadata(
            doc_id="doc3",
            text="Test",
            status=DocumentStatus.INDEXED,
            created_at=now,
            custom_metadata={"key": "value"},
        )
        result = doc.to_dict()

        assert result["doc_id"] == "doc3"
        assert result["status"] == "indexed"
        assert result["created_at"] == now.isoformat()
        assert result["custom_metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test DocumentMetadata deserialization."""
        now = datetime.now()
        data = {
            "doc_id": "doc4",
            "text": "Test text",
            "status": "indexed",
            "created_at": now.isoformat(),
            "word_count": 2,
            "custom_metadata": {"tags": ["a", "b"]},
        }
        doc = DocumentMetadata.from_dict(data)

        assert doc.doc_id == "doc4"
        assert doc.status == DocumentStatus.INDEXED
        assert doc.word_count == 2
        assert doc.custom_metadata["tags"] == ["a", "b"]


class TestDocumentStatus:
    """Tests for DocumentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.INDEXED.value == "indexed"
        assert DocumentStatus.FAILED.value == "failed"
        assert DocumentStatus.DELETED.value == "deleted"


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test QueryResult creation."""
        docs = [DocumentMetadata(doc_id="1", text="test")]
        result = QueryResult(
            documents=docs,
            total_count=10,
            query_time_ms=5.5,
            has_more=True,
        )
        assert len(result.documents) == 1
        assert result.total_count == 10
        assert result.has_more is True

    def test_query_result_to_dict(self):
        """Test QueryResult serialization."""
        docs = [DocumentMetadata(doc_id="1", text="test")]
        result = QueryResult(documents=docs, total_count=1, query_time_ms=1.0)
        data = result.to_dict()

        assert len(data["documents"]) == 1
        assert data["total_count"] == 1
        assert data["query_time_ms"] == 1.0


# ============================================================================
# DuckDBMetadataStore Initialization Tests
# ============================================================================


class TestDuckDBMetadataStoreInit:
    """Tests for DuckDBMetadataStore initialization."""

    def test_init_in_memory(self, reset_singleton):
        """Test in-memory initialization."""
        store = DuckDBMetadataStore(in_memory=True)
        assert store._in_memory is True
        assert store._db_path == ":memory:"

    def test_init_with_path(self, reset_singleton):
        """Test initialization with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            store = DuckDBMetadataStore(db_path=str(db_path))
            store.initialize()

            # Database should be created
            assert store._db_path == str(db_path)

            # Close the connection before directory cleanup
            store.close()

    def test_singleton_pattern(self, reset_singleton):
        """Test singleton pattern."""
        store1 = DuckDBMetadataStore(in_memory=True)
        store2 = DuckDBMetadataStore(in_memory=True)
        assert store1 is store2

    def test_initialize_creates_tables(self, store):
        """Test that initialize creates required tables."""
        # Check tables exist by querying them
        result = store.execute_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        table_names = {r["table_name"] for r in result}

        assert "documents" in table_names
        assert "ingestion_runs" in table_names
        assert "chunk_mappings" in table_names
        assert "index_versions" in table_names


# ============================================================================
# Document Operations Tests
# ============================================================================


class TestDocumentOperations:
    """Tests for document CRUD operations."""

    def test_add_single_document(self, store):
        """Test adding a single document."""
        doc = DocumentMetadata(
            doc_id="test_doc_1",
            text="This is a test document.",
            source_file="test.txt",
        )
        store.add_document(doc)

        # Retrieve and verify
        result = store.get_document("test_doc_1")
        assert result is not None
        assert result.doc_id == "test_doc_1"
        assert result.text == "This is a test document."

    def test_add_multiple_documents(self, store):
        """Test adding multiple documents."""
        docs = [DocumentMetadata(doc_id=f"doc_{i}", text=f"Document {i}") for i in range(5)]
        count = store.add_documents(docs)

        assert count == 5

        # Verify all were added
        results = store.get_documents([f"doc_{i}" for i in range(5)])
        assert len(results) == 5

    def test_add_document_auto_computes_counts(self, store):
        """Test that word/char counts are auto-computed."""
        doc = DocumentMetadata(
            doc_id="count_test",
            text="Hello world test",
        )
        store.add_document(doc)

        result = store.get_document("count_test")
        assert result.word_count == 3
        assert result.char_count == 16

    def test_get_nonexistent_document(self, store):
        """Test getting a document that doesn't exist."""
        result = store.get_document("nonexistent_id")
        assert result is None

    def test_update_document(self, store):
        """Test updating document metadata."""
        # Add document
        doc = DocumentMetadata(doc_id="update_test", text="Original text")
        store.add_document(doc)

        # Update
        store.update_document("update_test", {"status": DocumentStatus.INDEXED})

        # Verify
        result = store.get_document("update_test")
        assert result.status == DocumentStatus.INDEXED
        assert result.updated_at is not None

    def test_delete_document(self, store):
        """Test deleting a document."""
        doc = DocumentMetadata(doc_id="delete_test", text="To be deleted")
        store.add_document(doc)

        # Delete
        result = store.delete_document("delete_test")
        assert result is True

        # Verify deleted
        assert store.get_document("delete_test") is None

    def test_delete_multiple_documents(self, store):
        """Test deleting multiple documents."""
        docs = [DocumentMetadata(doc_id=f"del_{i}", text=f"Delete {i}") for i in range(3)]
        store.add_documents(docs)

        count = store.delete_documents(["del_0", "del_1", "del_2"])
        assert count == 3

    def test_mark_indexed(self, store):
        """Test marking documents as indexed."""
        docs = [DocumentMetadata(doc_id=f"idx_{i}", text=f"Index {i}") for i in range(3)]
        store.add_documents(docs)

        count = store.mark_indexed(["idx_0", "idx_1", "idx_2"])
        assert count == 3

        # Verify status updated
        result = store.get_document("idx_0")
        assert result.status == DocumentStatus.INDEXED
        assert result.indexed_at is not None


# ============================================================================
# Query Operations Tests
# ============================================================================


class TestQueryOperations:
    """Tests for query operations."""

    def test_query_all_documents(self, store):
        """Test querying all documents."""
        docs = [DocumentMetadata(doc_id=f"q_{i}", text=f"Query doc {i}") for i in range(10)]
        store.add_documents(docs)

        result = store.query()
        assert result.total_count == 10
        assert len(result.documents) == 10

    def test_query_with_where_filter(self, store):
        """Test querying with WHERE filter."""
        docs = [
            DocumentMetadata(
                doc_id=f"filter_{i}",
                text=f"Filter doc {i}",
                file_type="pdf" if i % 2 == 0 else "txt",
            )
            for i in range(10)
        ]
        store.add_documents(docs)

        result = store.query(where={"file_type": "pdf"})
        assert result.total_count == 5
        for doc in result.documents:
            assert doc.file_type == "pdf"

    def test_query_with_list_filter(self, store):
        """Test querying with IN filter."""
        docs = [DocumentMetadata(doc_id=f"list_{i}", text=f"List doc {i}") for i in range(5)]
        store.add_documents(docs)

        result = store.query(where={"doc_id": ["list_0", "list_2", "list_4"]})
        assert result.total_count == 3

    def test_query_with_comparison_operators(self, store):
        """Test querying with comparison operators."""
        docs = [
            DocumentMetadata(
                doc_id=f"cmp_{i}",
                text="x " * (i * 10),
                word_count=i * 10,
            )
            for i in range(1, 6)
        ]
        store.add_documents(docs)

        # Greater than
        result = store.query(where={"word_count": {"$gt": 20}})
        assert result.total_count == 3

        # Less than or equal
        result = store.query(where={"word_count": {"$lte": 30}})
        assert result.total_count == 3

    def test_query_with_limit_offset(self, store):
        """Test query pagination."""
        docs = [DocumentMetadata(doc_id=f"page_{i}", text=f"Page doc {i}") for i in range(25)]
        store.add_documents(docs)

        # First page
        result = store.query(limit=10, offset=0)
        assert len(result.documents) == 10
        assert result.total_count == 25
        assert result.has_more is True

        # Second page
        result = store.query(limit=10, offset=10)
        assert len(result.documents) == 10
        assert result.has_more is True

        # Last page
        result = store.query(limit=10, offset=20)
        assert len(result.documents) == 5
        assert result.has_more is False

    def test_query_with_order_by(self, store):
        """Test query ordering."""
        docs = [
            DocumentMetadata(
                doc_id=f"ord_{i}",
                text=f"Order doc {i}",
                word_count=i,
            )
            for i in range(5)
        ]
        store.add_documents(docs)

        # Ascending
        result = store.query(order_by="word_count")
        word_counts = [d.word_count for d in result.documents]
        assert word_counts == sorted(word_counts)

        # Descending
        result = store.query(order_by="-word_count")
        word_counts = [d.word_count for d in result.documents]
        assert word_counts == sorted(word_counts, reverse=True)

    def test_query_without_text(self, store):
        """Test query without including text."""
        doc = DocumentMetadata(
            doc_id="notext_test",
            text="This should not be returned",
        )
        store.add_document(doc)

        result = store.query(include_text=False)
        assert len(result.documents) == 1
        # Text should be empty string when not included
        assert result.documents[0].text == ""


# ============================================================================
# Full-Text Search Tests
# ============================================================================


class TestFullTextSearch:
    """Tests for full-text search."""

    def test_search_text_basic(self, store):
        """Test basic text search."""
        docs = [
            DocumentMetadata(doc_id="fts_1", text="The quick brown fox"),
            DocumentMetadata(doc_id="fts_2", text="The lazy dog"),
            DocumentMetadata(doc_id="fts_3", text="A quick rabbit"),
        ]
        store.add_documents(docs)

        result = store.search_text("quick")
        assert result.total_count >= 1  # At least one match

    def test_search_text_no_results(self, store):
        """Test text search with no matches."""
        docs = [
            DocumentMetadata(doc_id="fts_empty", text="Hello world"),
        ]
        store.add_documents(docs)

        result = store.search_text("nonexistent")
        assert result.total_count == 0

    def test_search_text_empty_query(self, store):
        """Test text search with empty query."""
        result = store.search_text("")
        assert result.total_count == 0


# ============================================================================
# Ingestion Run Tests
# ============================================================================


class TestIngestionRuns:
    """Tests for ingestion run operations."""

    def test_record_ingestion_run(self, store):
        """Test recording an ingestion run."""
        store.record_ingestion_run(
            run_id="run_001",
            source_folder="/data/docs",
            chunks_count=100,
            output_parquet="/output/chunks.parquet",
            metadata={"version": "1.0"},
        )

        result = store.get_ingestion_run("run_001")
        assert result is not None
        assert result["source_folder"] == "/data/docs"
        assert result["chunks_count"] == 100

    def test_update_ingestion_status(self, store):
        """Test updating ingestion status."""
        store.record_ingestion_run(
            run_id="run_002",
            source_folder="/data",
            chunks_count=50,
        )

        now = datetime.utcnow()
        store.update_ingestion_status(
            run_id="run_002",
            status="completed",
            started_at=now - timedelta(minutes=5),
            finished_at=now,
        )

        result = store.get_ingestion_run("run_002")
        assert result["status"] == "completed"


# ============================================================================
# Index Version Tests
# ============================================================================


class TestIndexVersions:
    """Tests for index version operations."""

    def test_record_index_version(self, store):
        """Test recording an index version."""
        store.record_index_version(
            version_id="v1.0.0",
            index_dir="/indexes/v1",
            document_count=1000,
            index_type="faiss",
            metadata={"dimension": 384},
        )

        result = store.get_latest_index_version()
        assert result is not None
        assert result["version_id"] == "v1.0.0"
        assert result["document_count"] == 1000

    def test_list_index_versions(self, store):
        """Test listing index versions."""
        for i in range(5):
            store.record_index_version(
                version_id=f"v{i}",
                index_dir=f"/indexes/v{i}",
                document_count=i * 100,
            )

        versions = store.list_index_versions(limit=3)
        assert len(versions) == 3

    def test_get_latest_by_type(self, store):
        """Test getting latest index by type."""
        store.record_index_version(
            version_id="faiss_v1",
            index_dir="/faiss",
            index_type="faiss",
        )
        store.record_index_version(
            version_id="bm25_v1",
            index_dir="/bm25",
            index_type="bm25",
        )

        result = store.get_latest_index_version(index_type="bm25")
        assert result["version_id"] == "bm25_v1"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for store statistics."""

    def test_get_statistics(self, store):
        """Test getting store statistics."""
        # Add some documents
        docs = [
            DocumentMetadata(
                doc_id=f"stat_{i}",
                text="Test " * 10,
                file_type="pdf" if i % 2 == 0 else "txt",
            )
            for i in range(10)
        ]
        store.add_documents(docs)
        store.mark_indexed(["stat_0", "stat_1"])

        stats = store.get_statistics()

        assert stats["total_documents"] == 10
        assert stats["documents_added"] == 10
        assert "documents_by_status" in stats
        assert "documents_by_file_type" in stats


# ============================================================================
# Raw SQL Query Tests
# ============================================================================


class TestRawSQLQuery:
    """Tests for raw SQL execution."""

    def test_execute_query(self, store):
        """Test executing raw SQL."""
        docs = [DocumentMetadata(doc_id=f"sql_{i}", text=f"SQL doc {i}") for i in range(5)]
        store.add_documents(docs)

        result = store.execute_query("SELECT COUNT(*) as cnt FROM documents")
        assert result[0]["cnt"] == 5

    def test_execute_query_with_params(self, store):
        """Test SQL with parameters."""
        docs = [DocumentMetadata(doc_id=f"param_{i}", text=f"Param doc {i}") for i in range(5)]
        store.add_documents(docs)

        result = store.execute_query("SELECT * FROM documents WHERE doc_id = ?", ["param_2"])
        assert len(result) == 1
        assert result[0]["doc_id"] == "param_2"


# ============================================================================
# Singleton Accessor Tests
# ============================================================================


class TestSingletonAccessor:
    """Tests for get_duckdb_metadata_store function."""

    def test_get_returns_singleton(self, reset_singleton):
        """Test that accessor returns singleton."""
        store1 = get_duckdb_metadata_store(in_memory=True)
        store2 = get_duckdb_metadata_store()

        assert store1 is store2

    def test_get_initializes_store(self, reset_singleton):
        """Test that accessor initializes the store."""
        store = get_duckdb_metadata_store(in_memory=True)

        # Should be able to add documents
        doc = DocumentMetadata(doc_id="singleton_test", text="Test")
        store.add_document(doc)

        result = store.get_document("singleton_test")
        assert result is not None


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_documents_list(self, store):
        """Test adding empty documents list."""
        count = store.add_documents([])
        assert count == 0

    def test_empty_doc_ids_list(self, store):
        """Test getting empty doc IDs list."""
        results = store.get_documents([])
        assert results == []

    def test_update_nonexistent_document(self, store):
        """Test updating a document that doesn't exist."""
        result = store.update_document("nonexistent", {"status": "indexed"})
        # Should not raise, just return True (DuckDB doesn't tell us rows affected easily)
        assert result is True

    def test_document_with_special_characters(self, store):
        """Test documents with special characters."""
        doc = DocumentMetadata(
            doc_id="special_chars",
            text="Text with 'quotes' and \"double quotes\" and emoji 🎉",
            custom_metadata={"key": "value with 'quotes'"},
        )
        store.add_document(doc)

        result = store.get_document("special_chars")
        assert "emoji" in result.text

    def test_document_with_long_text(self, store):
        """Test documents with very long text."""
        long_text = "word " * 10000
        doc = DocumentMetadata(doc_id="long_text", text=long_text)
        store.add_document(doc)

        result = store.get_document("long_text")
        assert len(result.text) == len(long_text)

    def test_concurrent_operations(self, reset_singleton):
        """Test concurrent document operations with file-based store."""
        import threading

        # Use file-based store for concurrent access
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "concurrent.duckdb"
            store = DuckDBMetadataStore(db_path=str(db_path))
            store.initialize()

            errors = []

            def add_docs(start_id):
                try:
                    for i in range(10):
                        doc = DocumentMetadata(
                            doc_id=f"concurrent_{start_id}_{i}",
                            text=f"Concurrent doc {i}",
                        )
                        store.add_document(doc)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_docs, args=(i,)) for i in range(3)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

            # Verify all documents were added
            result = store.query(where={"doc_id": {"$like": "concurrent_%"}})
            assert result.total_count == 30

            # Close before cleanup
            store.close()
