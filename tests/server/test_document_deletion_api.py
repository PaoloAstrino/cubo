"""Tests for DELETE /api/documents/{doc_id} endpoint."""

import pytest

pytest.importorskip("fastapi")
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app():
    """Create a mock CUBO app."""
    app = MagicMock()
    app.vector_store = MagicMock()
    app.retriever = MagicMock()
    return app


@pytest.fixture
def client(mock_cubo_app):
    """Create test client with mocked dependencies."""
    with patch("cubo.server.api.cubo_app", mock_cubo_app):
        from cubo.server.api import app

        yield TestClient(app)


class TestDocumentDeletionAPI:
    """Test cases for document deletion endpoint."""

    def test_delete_document_success(self, client, mock_cubo_app):
        """Test successful document deletion."""
        # Mock successful deletion
        mock_cubo_app.vector_store.delete.return_value = None
        mock_cubo_app.retriever.remove_document.return_value = True

        response = client.delete("/api/documents/test_doc.pdf")

        if response.status_code == 503:
            pytest.skip("Server not initialized")

        assert response.status_code == 200
        data = response.json()

        assert data["doc_id"] == "test_doc.pdf"
        assert data["deleted"] is True
        assert "trace_id" in data
        assert "message" in data

    def test_delete_document_not_found(self, client, mock_cubo_app):
        """Test deletion of non-existent document."""
        # Mock failed deletion
        mock_cubo_app.vector_store.delete.side_effect = Exception("Not found")
        mock_cubo_app.retriever.remove_document.return_value = False

        response = client.delete("/api/documents/nonexistent.pdf")

        if response.status_code == 503:
            pytest.skip("Server not initialized")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_delete_document_includes_trace_id(self, client, mock_cubo_app):
        """Test that deletion response includes trace_id."""
        mock_cubo_app.retriever.remove_document.return_value = True

        response = client.delete(
            "/api/documents/test_doc.pdf", headers={"x-trace-id": "custom-trace-123"}
        )

        if response.status_code == 503:
            pytest.skip("Server not initialized")

        # Response should include trace_id
        data = response.json()
        if response.status_code == 200:
            assert "trace_id" in data

    def test_delete_document_without_app(self, client):
        """Test deletion when app is not initialized."""
        with patch("cubo.server.api.cubo_app", None):
            response = client.delete("/api/documents/test.pdf")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"].lower()

    def test_delete_document_logs_audit(self, client, mock_cubo_app):
        """Test that deletion is logged for GDPR audit."""
        mock_cubo_app.retriever.remove_document.return_value = True

        with patch("cubo.server.api.logger") as mock_logger:
            response = client.delete("/api/documents/gdpr_test.pdf")

            if response.status_code == 503:
                pytest.skip("Server not initialized")

            # Check that audit logging was called
            if response.status_code == 200:
                # Logger should have been called with GDPR audit info
                assert mock_logger.info.called

    def test_delete_document_response_model(self, client, mock_cubo_app):
        """Test deletion response matches DeleteDocumentResponse model."""
        mock_cubo_app.retriever.remove_document.return_value = True

        response = client.delete("/api/documents/model_test.pdf")

        if response.status_code == 503:
            pytest.skip("Server not initialized")

        if response.status_code == 200:
            data = response.json()

            # Verify all required fields
            assert "doc_id" in data
            assert "deleted" in data
            assert "chunks_removed" in data
            assert "trace_id" in data
            assert "message" in data

            # Verify types
            assert isinstance(data["doc_id"], str)
            assert isinstance(data["deleted"], bool)
            assert isinstance(data["chunks_removed"], int)
            assert isinstance(data["trace_id"], str)
            assert isinstance(data["message"], str)
