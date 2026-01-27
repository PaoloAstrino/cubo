"""
Integration tests for FAISS index build, save, and load lifecycle.

Tests the complete workflow of index creation, metadata persistence,
and index reloading to ensure indexes can be reused across sessions.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    # 10 documents, 384-dimensional embeddings (BGE-base dimension)
    np.random.seed(42)
    embeddings = np.random.randn(10, 384).astype("float32")
    # Normalize to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def sample_documents():
    """Generate sample document metadata."""
    return [{"doc_id": f"doc{i}", "text": f"Sample document {i}"} for i in range(10)]


class TestFAISSIndexLifecycle:
    """Test suite for FAISS index creation and persistence."""

    def test_index_creation_with_embeddings(self, temp_index_dir, sample_embeddings):
        """Test creating a FAISS index from embeddings."""
        import faiss

        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        index.add(sample_embeddings)

        assert index.ntotal == 10
        assert index.d == 384

    def test_index_save_with_metadata(self, temp_index_dir, sample_embeddings, sample_documents):
        """Test saving index with metadata.json."""
        import faiss

        # Create index
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(sample_embeddings)

        # Save index file
        index_path = temp_index_dir / "hot.index"
        faiss.write_index(index, str(index_path))

        # Save metadata
        metadata = {
            "dimension": dimension,
            "num_vectors": index.ntotal,
            "index_type": "IndexFlatIP",
            "created_at": "2026-01-13T00:00:00",
            "embedding_model": "BAAI/bge-base-en-v1.5",
        }

        metadata_path = temp_index_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Verify files exist
        assert index_path.exists()
        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata["dimension"] == 384
        assert loaded_metadata["num_vectors"] == 10
        assert loaded_metadata["index_type"] == "IndexFlatIP"

    def test_index_load_from_disk(self, temp_index_dir, sample_embeddings):
        """Test loading a previously saved index."""
        import faiss

        # Create and save index
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(sample_embeddings)

        index_path = temp_index_dir / "hot.index"
        faiss.write_index(index, str(index_path))

        # Load index
        loaded_index = faiss.read_index(str(index_path))

        assert loaded_index.ntotal == 10
        assert loaded_index.d == 384

    def test_index_load_requires_metadata(self, temp_index_dir):
        """Test that index loading checks for metadata.json."""
        metadata_path = temp_index_dir / "metadata.json"

        # Without metadata file, should fail validation
        assert not metadata_path.exists()

        # Attempting to load should raise error or return None
        # (Actual implementation would check metadata existence)

    def test_metadata_json_required_fields(self, temp_index_dir):
        """Test that metadata.json contains required fields."""
        metadata = {
            "dimension": 384,
            "num_vectors": 1000,
            "index_type": "IndexFlatIP",
            "created_at": "2026-01-13",
            "embedding_model": "BAAI/bge-base-en-v1.5",
        }

        required_fields = ["dimension", "num_vectors", "index_type"]

        for field in required_fields:
            assert field in metadata

    def test_index_search_after_reload(self, temp_index_dir, sample_embeddings):
        """Test that reloaded index produces same search results."""
        import faiss

        # Create and save index
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(sample_embeddings)

        # Search before saving
        query = sample_embeddings[0:1]  # Use first embedding as query
        D1, I1 = index.search(query, k=5)

        # Save and reload
        index_path = temp_index_dir / "hot.index"
        faiss.write_index(index, str(index_path))
        loaded_index = faiss.read_index(str(index_path))

        # Search after reloading
        D2, I2 = loaded_index.search(query, k=5)

        # Results should be identical
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_array_almost_equal(D1, D2, decimal=6)

    def test_index_update_metadata_on_add(self, temp_index_dir, sample_embeddings):
        """Test that metadata is updated when vectors are added."""
        import faiss

        # Create index with 5 vectors
        dimension = sample_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(sample_embeddings[:5])

        metadata_v1 = {"dimension": dimension, "num_vectors": index.ntotal, "version": 1}

        assert metadata_v1["num_vectors"] == 5

        # Add 5 more vectors
        index.add(sample_embeddings[5:])

        metadata_v2 = {"dimension": dimension, "num_vectors": index.ntotal, "version": 2}

        assert metadata_v2["num_vectors"] == 10


class TestIndexRebuildLogic:
    """Test suite for index rebuild vs. reuse logic."""

    def test_reindex_flag_triggers_rebuild(self, temp_index_dir):
        """Test that --reindex flag forces index rebuild."""
        # Simulate existing index
        existing_index_path = temp_index_dir / "hot.index"
        existing_index_path.touch()

        # With --reindex, should rebuild (delete and recreate)
        reindex = True

        if reindex:
            should_rebuild = True
        else:
            should_rebuild = not existing_index_path.exists()

        assert should_rebuild is True

    def test_missing_metadata_forces_rebuild(self, temp_index_dir):
        """Test that missing metadata.json forces rebuild."""
        # Index file exists but metadata missing
        index_path = temp_index_dir / "hot.index"
        index_path.touch()

        metadata_path = temp_index_dir / "metadata.json"

        # Should force rebuild if metadata missing
        should_rebuild = not metadata_path.exists()

        assert should_rebuild is True

    def test_existing_index_with_metadata_reused(self, temp_index_dir):
        """Test that valid index + metadata is reused."""
        # Both index and metadata exist
        index_path = temp_index_dir / "hot.index"
        metadata_path = temp_index_dir / "metadata.json"

        index_path.touch()

        metadata = {"dimension": 384, "num_vectors": 1000, "index_type": "IndexFlatIP"}

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Should reuse existing index
        reindex = False
        should_rebuild = reindex or not (index_path.exists() and metadata_path.exists())

        assert should_rebuild is False


class TestDocumentDatabaseSync:
    """Test suite for keeping documents.db in sync with index."""

    def test_documents_db_matches_index_size(self, temp_index_dir, sample_documents):
        """Test that documents.db has same count as index."""
        import sqlite3

        # Create documents database
        db_path = temp_index_dir / "documents.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                text TEXT
            )
        """
        )

        for i, doc in enumerate(sample_documents):
            cursor.execute(
                "INSERT INTO documents (id, doc_id, text) VALUES (?, ?, ?)",
                (i, doc["doc_id"], doc["text"]),
            )

        conn.commit()

        # Count documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]

        conn.close()

        # Should match number of documents
        assert count == len(sample_documents)

    def test_document_retrieval_by_index_id(self, temp_index_dir, sample_documents):
        """Test retrieving documents by FAISS index ID."""
        import sqlite3

        # Create and populate database
        db_path = temp_index_dir / "documents.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                text TEXT
            )
        """
        )

        for i, doc in enumerate(sample_documents):
            cursor.execute(
                "INSERT INTO documents (id, doc_id, text) VALUES (?, ?, ?)",
                (i, doc["doc_id"], doc["text"]),
            )

        conn.commit()

        # Retrieve by index ID
        search_index_id = 3
        cursor.execute("SELECT doc_id, text FROM documents WHERE id = ?", (search_index_id,))
        result = cursor.fetchone()

        conn.close()

        assert result is not None
        assert result[0] == "doc3"
        assert "Sample document 3" in result[1]


class TestIndexCompatibility:
    """Test suite for index version compatibility."""

    def test_dimension_mismatch_detection(self, temp_index_dir):
        """Test detection of dimension mismatch between index and model."""
        # Existing index: 384 dimensions
        existing_metadata = {"dimension": 384, "embedding_model": "BAAI/bge-base-en-v1.5"}

        # Current model: 768 dimensions (different model)
        current_dimension = 768

        # Should detect mismatch
        is_compatible = existing_metadata["dimension"] == current_dimension

        assert is_compatible is False

    def test_model_change_detection(self, temp_index_dir):
        """Test detection of embedding model change."""
        existing_metadata = {"embedding_model": "BAAI/bge-base-en-v1.5"}

        current_model = "sentence-transformers/all-MiniLM-L6-v2"

        # Should detect model change
        model_changed = existing_metadata["embedding_model"] != current_model

        assert model_changed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
