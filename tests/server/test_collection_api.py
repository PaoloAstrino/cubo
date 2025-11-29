"""Integration tests for collection API endpoints."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app():
    """Create a mock CUBOApp with vector_store."""
    mock_app = MagicMock()
    mock_vector_store = MagicMock()
    mock_app.vector_store = mock_vector_store
    return mock_app, mock_vector_store


@pytest.fixture
def client(mock_cubo_app):
    """Create test client with mocked CUBO app."""
    mock_app, mock_vector_store = mock_cubo_app
    
    with patch("src.cubo.server.api.cubo_app", mock_app):
        from src.cubo.server.api import app
        yield TestClient(app), mock_vector_store


class TestListCollections:
    """Test GET /api/collections endpoint."""

    def test_list_collections_empty(self, client):
        """Test listing collections when none exist."""
        test_client, mock_store = client
        mock_store.list_collections.return_value = []
        
        response = test_client.get("/api/collections")
        
        assert response.status_code == 200
        assert response.json() == []
        mock_store.list_collections.assert_called_once()

    def test_list_collections_with_data(self, client):
        """Test listing collections with data."""
        test_client, mock_store = client
        mock_store.list_collections.return_value = [
            {
                "id": "coll-1",
                "name": "Research Papers",
                "color": "#2563eb",
                "created_at": "2025-11-29T10:00:00",
                "document_count": 5
            },
            {
                "id": "coll-2",
                "name": "Project Docs",
                "color": "#10b981",
                "created_at": "2025-11-29T11:00:00",
                "document_count": 3
            }
        ]
        
        response = test_client.get("/api/collections")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Research Papers"
        assert data[1]["name"] == "Project Docs"


class TestCreateCollection:
    """Test POST /api/collections endpoint."""

    def test_create_collection_success(self, client):
        """Test successful collection creation."""
        test_client, mock_store = client
        mock_store.create_collection.return_value = {
            "id": "new-coll-id",
            "name": "New Collection",
            "color": "#ff0000",
            "created_at": "2025-11-29T12:00:00",
            "document_count": 0
        }
        
        response = test_client.post(
            "/api/collections",
            json={"name": "New Collection", "color": "#ff0000"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Collection"
        assert data["color"] == "#ff0000"
        assert data["id"] == "new-coll-id"
        mock_store.create_collection.assert_called_once_with(
            name="New Collection",
            color="#ff0000"
        )

    def test_create_collection_default_color(self, client):
        """Test collection creation with default color."""
        test_client, mock_store = client
        mock_store.create_collection.return_value = {
            "id": "coll-id",
            "name": "Default Color",
            "color": "#2563eb",
            "created_at": "2025-11-29T12:00:00",
            "document_count": 0
        }
        
        response = test_client.post(
            "/api/collections",
            json={"name": "Default Color"}
        )
        
        assert response.status_code == 200
        assert response.json()["color"] == "#2563eb"

    def test_create_duplicate_collection_fails(self, client):
        """Test that duplicate collection names return 409."""
        test_client, mock_store = client
        mock_store.create_collection.side_effect = ValueError("Collection 'Duplicate' already exists")
        
        response = test_client.post(
            "/api/collections",
            json={"name": "Duplicate"}
        )
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_collection_empty_name_fails(self, client):
        """Test that empty name is rejected."""
        test_client, _ = client
        
        response = test_client.post(
            "/api/collections",
            json={"name": ""}
        )
        
        assert response.status_code == 422  # Validation error


class TestGetCollection:
    """Test GET /api/collections/{collection_id} endpoint."""

    def test_get_collection_success(self, client):
        """Test getting a specific collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = {
            "id": "coll-123",
            "name": "Specific Collection",
            "color": "#2563eb",
            "created_at": "2025-11-29T12:00:00",
            "document_count": 10
        }
        
        response = test_client.get("/api/collections/coll-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "coll-123"
        assert data["name"] == "Specific Collection"
        mock_store.get_collection.assert_called_once_with("coll-123")

    def test_get_collection_not_found(self, client):
        """Test getting a non-existent collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = None
        
        response = test_client.get("/api/collections/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDeleteCollection:
    """Test DELETE /api/collections/{collection_id} endpoint."""

    def test_delete_collection_success(self, client):
        """Test successful collection deletion."""
        test_client, mock_store = client
        mock_store.delete_collection.return_value = True
        
        response = test_client.delete("/api/collections/coll-to-delete")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["collection_id"] == "coll-to-delete"

    def test_delete_collection_not_found(self, client):
        """Test deleting non-existent collection."""
        test_client, mock_store = client
        mock_store.delete_collection.return_value = False
        
        response = test_client.delete("/api/collections/nonexistent")
        
        assert response.status_code == 404


class TestAddDocumentsToCollection:
    """Test POST /api/collections/{collection_id}/documents endpoint."""

    def test_add_documents_success(self, client):
        """Test adding documents to a collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = {"id": "coll-1", "name": "Test"}
        mock_store.add_documents_to_collection.return_value = {
            "added_count": 3,
            "already_in_collection": 0
        }
        
        response = test_client.post(
            "/api/collections/coll-1/documents",
            json={"document_ids": ["doc1", "doc2", "doc3"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["added_count"] == 3
        assert data["already_in_collection"] == 0

    def test_add_documents_collection_not_found(self, client):
        """Test adding documents to non-existent collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = None
        
        response = test_client.post(
            "/api/collections/nonexistent/documents",
            json={"document_ids": ["doc1"]}
        )
        
        assert response.status_code == 404


class TestRemoveDocumentsFromCollection:
    """Test DELETE /api/collections/{collection_id}/documents endpoint."""

    def test_remove_documents_success(self, client):
        """Test removing documents from a collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = {"id": "coll-1", "name": "Test"}
        mock_store.remove_documents_from_collection.return_value = 2
        
        response = test_client.request(
            "DELETE",
            "/api/collections/coll-1/documents",
            json={"document_ids": ["doc1", "doc2"]}
        )
        
        assert response.status_code == 200
        assert response.json()["removed_count"] == 2


class TestGetCollectionDocuments:
    """Test GET /api/collections/{collection_id}/documents endpoint."""

    def test_get_collection_documents_success(self, client):
        """Test getting documents from a collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = {"id": "coll-1", "name": "Test"}
        mock_store.get_collection_documents.return_value = ["doc1", "doc2", "doc3"]
        
        response = test_client.get("/api/collections/coll-1/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["collection_id"] == "coll-1"
        assert data["document_ids"] == ["doc1", "doc2", "doc3"]
        assert data["count"] == 3

    def test_get_collection_documents_not_found(self, client):
        """Test getting documents from non-existent collection."""
        test_client, mock_store = client
        mock_store.get_collection.return_value = None
        
        response = test_client.get("/api/collections/nonexistent/documents")
        
        assert response.status_code == 404
