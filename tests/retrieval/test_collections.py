"""Tests for document collection management in FaissStore."""

import json
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("torch")

from cubo.retrieval.vector_store import FaissStore


@pytest.fixture
def store(tmp_path: Path) -> FaissStore:
    """Create a FaissStore instance with test data."""
    dim = 8
    store = FaissStore(dimension=dim, index_dir=tmp_path / "faiss_index")
    
    # Add test documents
    ids = [f"doc{i}" for i in range(5)]
    vectors = [np.ones(dim) * (i + 1) for i in range(5)]
    docs = [f"Document {i} content" for i in range(5)]
    metas = [{"filename": f"file{i}.txt", "page": i} for i in range(5)]
    store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)
    
    return store


class TestCollectionCRUD:
    """Test collection create, read, update, delete operations."""

    def test_create_collection(self, store: FaissStore):
        """Test creating a new collection."""
        collection = store.create_collection(name="Test Collection", color="#ff0000")
        
        assert collection["name"] == "Test Collection"
        assert collection["color"] == "#ff0000"
        assert collection["document_count"] == 0
        assert "id" in collection
        assert "created_at" in collection

    def test_create_collection_default_color(self, store: FaissStore):
        """Test creating a collection with default color."""
        collection = store.create_collection(name="Default Color")
        
        assert collection["color"] == "#2563eb"  # Brand blue

    def test_create_duplicate_collection_fails(self, store: FaissStore):
        """Test that creating a duplicate collection raises ValueError."""
        store.create_collection(name="Unique Name")
        
        with pytest.raises(ValueError, match="already exists"):
            store.create_collection(name="Unique Name")

    def test_list_collections_empty(self, store: FaissStore):
        """Test listing collections when none exist."""
        collections = store.list_collections()
        assert collections == []

    def test_list_collections(self, store: FaissStore):
        """Test listing multiple collections."""
        store.create_collection(name="Collection A")
        store.create_collection(name="Collection B")
        store.create_collection(name="Collection C")
        
        collections = store.list_collections()
        
        assert len(collections) == 3
        names = [c["name"] for c in collections]
        assert "Collection A" in names
        assert "Collection B" in names
        assert "Collection C" in names

    def test_get_collection(self, store: FaissStore):
        """Test getting a specific collection by ID."""
        created = store.create_collection(name="Get Test")
        
        retrieved = store.get_collection(created["id"])
        
        assert retrieved is not None
        assert retrieved["id"] == created["id"]
        assert retrieved["name"] == "Get Test"

    def test_get_nonexistent_collection(self, store: FaissStore):
        """Test getting a collection that doesn't exist."""
        result = store.get_collection("nonexistent-id")
        assert result is None

    def test_delete_collection(self, store: FaissStore):
        """Test deleting a collection."""
        collection = store.create_collection(name="To Delete")
        
        result = store.delete_collection(collection["id"])
        
        assert result is True
        assert store.get_collection(collection["id"]) is None

    def test_delete_nonexistent_collection(self, store: FaissStore):
        """Test deleting a collection that doesn't exist."""
        result = store.delete_collection("nonexistent-id")
        assert result is False


class TestCollectionDocuments:
    """Test adding and removing documents from collections."""

    def test_add_documents_to_collection(self, store: FaissStore):
        """Test adding documents to a collection."""
        collection = store.create_collection(name="With Docs")
        
        result = store.add_documents_to_collection(
            collection["id"], 
            ["doc0", "doc1", "doc2"]
        )
        
        assert result["added_count"] == 3
        assert result["already_in_collection"] == 0
        
        # Verify document count updated
        updated = store.get_collection(collection["id"])
        assert updated["document_count"] == 3

    def test_add_duplicate_documents(self, store: FaissStore):
        """Test adding documents that are already in the collection."""
        collection = store.create_collection(name="Duplicates")
        store.add_documents_to_collection(collection["id"], ["doc0", "doc1"])
        
        # Add doc1 again plus new doc2
        result = store.add_documents_to_collection(
            collection["id"], 
            ["doc1", "doc2"]
        )
        
        assert result["added_count"] == 1  # Only doc2 added
        assert result["already_in_collection"] == 1  # doc1 already there

    def test_get_collection_documents(self, store: FaissStore):
        """Test retrieving document IDs from a collection."""
        collection = store.create_collection(name="Get Docs")
        store.add_documents_to_collection(collection["id"], ["doc0", "doc2", "doc4"])
        
        doc_ids = store.get_collection_documents(collection["id"])
        
        assert len(doc_ids) == 3
        assert set(doc_ids) == {"doc0", "doc2", "doc4"}

    def test_remove_documents_from_collection(self, store: FaissStore):
        """Test removing documents from a collection."""
        collection = store.create_collection(name="Remove Docs")
        store.add_documents_to_collection(
            collection["id"], 
            ["doc0", "doc1", "doc2", "doc3"]
        )
        
        removed = store.remove_documents_from_collection(
            collection["id"], 
            ["doc1", "doc3"]
        )
        
        assert removed == 2
        
        remaining = store.get_collection_documents(collection["id"])
        assert set(remaining) == {"doc0", "doc2"}

    def test_remove_nonexistent_documents(self, store: FaissStore):
        """Test removing documents that aren't in the collection."""
        collection = store.create_collection(name="Remove Missing")
        store.add_documents_to_collection(collection["id"], ["doc0"])
        
        removed = store.remove_documents_from_collection(
            collection["id"], 
            ["doc5", "doc6"]  # Don't exist
        )
        
        assert removed == 0

    def test_get_document_filenames_in_collection(self, store: FaissStore):
        """Test getting filenames from documents in a collection."""
        collection = store.create_collection(name="Filenames")
        store.add_documents_to_collection(
            collection["id"], 
            ["doc0", "doc1", "doc2"]
        )
        
        filenames = store.get_document_filenames_in_collection(collection["id"])
        
        assert len(filenames) == 3
        assert set(filenames) == {"file0.txt", "file1.txt", "file2.txt"}


class TestCollectionDeletion:
    """Test collection deletion behavior."""

    def test_delete_collection_removes_document_links(self, store: FaissStore):
        """Test that deleting a collection removes document associations."""
        collection = store.create_collection(name="Delete With Docs")
        store.add_documents_to_collection(collection["id"], ["doc0", "doc1"])
        
        store.delete_collection(collection["id"])
        
        # Collection should be gone
        assert store.get_collection(collection["id"]) is None
        
        # Documents should still exist in store (not deleted, just unlinked)
        doc = store._get_document_from_db("doc0")
        assert doc is not None

    def test_documents_can_belong_to_multiple_collections(self, store: FaissStore):
        """Test that a document can be in multiple collections."""
        coll_a = store.create_collection(name="Collection A")
        coll_b = store.create_collection(name="Collection B")
        
        store.add_documents_to_collection(coll_a["id"], ["doc0", "doc1"])
        store.add_documents_to_collection(coll_b["id"], ["doc1", "doc2"])
        
        # doc1 should be in both
        docs_a = store.get_collection_documents(coll_a["id"])
        docs_b = store.get_collection_documents(coll_b["id"])
        
        assert "doc1" in docs_a
        assert "doc1" in docs_b
        
        # Deleting collection A shouldn't affect B
        store.delete_collection(coll_a["id"])
        docs_b_after = store.get_collection_documents(coll_b["id"])
        assert "doc1" in docs_b_after


class TestCollectionQueryFiltering:
    """Test using collections to filter queries."""

    def test_get_filenames_for_query_filtering(self, store: FaissStore):
        """Test getting filenames to use as query filter."""
        collection = store.create_collection(name="Query Filter")
        store.add_documents_to_collection(collection["id"], ["doc0", "doc2"])
        
        filenames = store.get_document_filenames_in_collection(collection["id"])
        
        # These can be used in query's where clause
        assert "file0.txt" in filenames
        assert "file2.txt" in filenames
        assert "file1.txt" not in filenames
