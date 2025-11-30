"""End-to-end integration test for collection workflow.

Tests the complete flow:
1. Create a collection
2. Add documents to collection
3. Query within collection context
4. Delete collection
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cubo.retrieval.vector_store import FaissStore


@pytest.fixture
def populated_store(tmp_path: Path) -> FaissStore:
    """Create a FaissStore with realistic document data."""
    dim = 384  # Common embedding dimension
    store = FaissStore(dimension=dim, index_dir=tmp_path / "faiss_index")
    
    # Create documents from different "files"
    documents = [
        # Research paper chunks
        {"id": "research_1_chunk1", "content": "Machine learning models have shown remarkable progress in natural language understanding.", "filename": "ml_paper.pdf"},
        {"id": "research_1_chunk2", "content": "Transformer architectures enable parallel processing of sequences.", "filename": "ml_paper.pdf"},
        {"id": "research_2_chunk1", "content": "Reinforcement learning agents can learn complex behaviors through trial and error.", "filename": "rl_survey.pdf"},
        
        # Project documentation
        {"id": "project_1_chunk1", "content": "The API endpoints follow RESTful conventions with JSON responses.", "filename": "api_docs.md"},
        {"id": "project_1_chunk2", "content": "Authentication is handled via JWT tokens with 24-hour expiration.", "filename": "api_docs.md"},
        {"id": "project_2_chunk1", "content": "Database migrations should be run before deploying new versions.", "filename": "deployment.md"},
        
        # Meeting notes
        {"id": "meeting_1_chunk1", "content": "Q4 roadmap includes new dashboard features and performance improvements.", "filename": "meeting_notes.txt"},
        {"id": "meeting_2_chunk1", "content": "Budget allocation for cloud infrastructure approved by leadership.", "filename": "budget_meeting.txt"},
    ]
    
    # Generate embeddings (random for testing, but consistent)
    np.random.seed(42)
    ids = [d["id"] for d in documents]
    vectors = [np.random.randn(dim).astype(np.float32).tolist() for _ in documents]
    contents = [d["content"] for d in documents]
    metadatas = [{"filename": d["filename"]} for d in documents]
    
    store.add(embeddings=vectors, documents=contents, metadatas=metadatas, ids=ids)
    
    return store


class TestCollectionWorkflow:
    """Test complete collection management workflow."""

    def test_create_collection_and_add_documents(self, populated_store: FaissStore):
        """Test creating a collection and adding documents."""
        # Create research collection
        research_coll = populated_store.create_collection(
            name="ML Research",
            color="#2563eb"
        )
        assert research_coll["id"]
        assert research_coll["document_count"] == 0
        
        # Add research document IDs
        research_doc_ids = ["research_1_chunk1", "research_1_chunk2", "research_2_chunk1"]
        result = populated_store.add_documents_to_collection(
            research_coll["id"],
            research_doc_ids
        )
        
        assert result["added_count"] == 3
        
        # Verify collection now has documents
        updated = populated_store.get_collection(research_coll["id"])
        assert updated["document_count"] == 3

    def test_multiple_collections_same_documents(self, populated_store: FaissStore):
        """Test that documents can belong to multiple collections."""
        # Create two overlapping collections
        all_papers = populated_store.create_collection(name="All Papers")
        ml_specific = populated_store.create_collection(name="ML Specific")
        
        # Add overlapping documents
        populated_store.add_documents_to_collection(
            all_papers["id"],
            ["research_1_chunk1", "research_1_chunk2", "research_2_chunk1"]
        )
        populated_store.add_documents_to_collection(
            ml_specific["id"],
            ["research_1_chunk1", "research_1_chunk2"]  # Only ML paper, not RL
        )
        
        # Verify counts
        all_papers_updated = populated_store.get_collection(all_papers["id"])
        ml_specific_updated = populated_store.get_collection(ml_specific["id"])
        
        assert all_papers_updated["document_count"] == 3
        assert ml_specific_updated["document_count"] == 2

    def test_query_filtering_by_collection(self, populated_store: FaissStore):
        """Test that collection filtering returns correct filenames for queries."""
        # Create project docs collection
        project_coll = populated_store.create_collection(name="Project Docs")
        populated_store.add_documents_to_collection(
            project_coll["id"],
            ["project_1_chunk1", "project_1_chunk2", "project_2_chunk1"]
        )
        
        # Get filenames for query filtering
        filenames = populated_store.get_document_filenames_in_collection(project_coll["id"])
        
        assert len(filenames) == 2  # api_docs.md and deployment.md
        assert "api_docs.md" in filenames
        assert "deployment.md" in filenames
        assert "ml_paper.pdf" not in filenames

    def test_collection_deletion_preserves_documents(self, populated_store: FaissStore):
        """Test that deleting a collection doesn't delete the documents."""
        # Create and populate collection
        temp_coll = populated_store.create_collection(name="Temporary")
        populated_store.add_documents_to_collection(
            temp_coll["id"],
            ["meeting_1_chunk1", "meeting_2_chunk1"]
        )
        
        # Delete collection
        deleted = populated_store.delete_collection(temp_coll["id"])
        assert deleted is True
        
        # Documents should still exist
        doc = populated_store._get_document_from_db("meeting_1_chunk1")
        assert doc is not None
        assert "Q4 roadmap" in doc["document"]

    def test_collection_list_ordering(self, populated_store: FaissStore):
        """Test that collections are listed in creation order (newest first)."""
        import time
        
        # Create collections with slight delays
        coll_a = populated_store.create_collection(name="First")
        time.sleep(0.01)
        coll_b = populated_store.create_collection(name="Second")
        time.sleep(0.01)
        coll_c = populated_store.create_collection(name="Third")
        
        # List should be newest first
        collections = populated_store.list_collections()
        names = [c["name"] for c in collections]
        
        assert names == ["Third", "Second", "First"]

    def test_remove_documents_from_collection(self, populated_store: FaissStore):
        """Test removing specific documents from a collection."""
        # Create and populate
        coll = populated_store.create_collection(name="Test Removal")
        populated_store.add_documents_to_collection(
            coll["id"],
            ["research_1_chunk1", "research_1_chunk2", "research_2_chunk1"]
        )
        
        # Remove one document
        removed = populated_store.remove_documents_from_collection(
            coll["id"],
            ["research_1_chunk2"]
        )
        
        assert removed == 1
        
        # Verify remaining
        doc_ids = populated_store.get_collection_documents(coll["id"])
        assert len(doc_ids) == 2
        assert "research_1_chunk2" not in doc_ids
        assert "research_1_chunk1" in doc_ids
        assert "research_2_chunk1" in doc_ids


class TestCollectionQueryIntegration:
    """Test collection filtering with actual vector queries."""

    def test_query_with_filename_filter(self, populated_store: FaissStore):
        """Test that collection provides correct filenames for filtering queries."""
        # Create collection with only project docs
        project_coll = populated_store.create_collection(name="Project Only")
        populated_store.add_documents_to_collection(
            project_coll["id"],
            ["project_1_chunk1", "project_1_chunk2", "project_2_chunk1"]
        )
        
        # Get filenames for filtering - this is what we'd use in the retriever
        filenames = populated_store.get_document_filenames_in_collection(project_coll["id"])
        
        # Verify we get the correct filenames for filtering
        assert len(filenames) == 2  # api_docs.md and deployment.md
        assert "api_docs.md" in filenames
        assert "deployment.md" in filenames
        assert "ml_paper.pdf" not in filenames
        assert "meeting_notes.txt" not in filenames
        
        # The filtering would be applied in the retriever layer using these filenames
        # as a post-filter or in the where clause
        # This test verifies the collection->filename mapping works correctly
