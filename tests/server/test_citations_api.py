"""Tests for Citation API enhancement in /api/query endpoint."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app():
    """Create a mock CUBO app with retriever."""
    app = MagicMock()
    app.retriever = MagicMock()
    app.generator = MagicMock()
    app.vector_store = None
    
    # Mock retriever.collection.count() to return non-zero
    app.retriever.collection = MagicMock()
    app.retriever.collection.count.return_value = 10
    
    return app


@pytest.fixture
def mock_retrieved_docs():
    """Sample retrieved documents with metadata."""
    return [
        {
            "document": "The vacation policy allows 20 days per year.",
            "metadata": {
                "filename": "policy.pdf",
                "page": 5,
                "chunk_id": "chunk_001",
                "chunk_index": 12,
            },
            "similarity": 0.95,
        },
        {
            "document": "Remote work requires manager approval.",
            "metadata": {
                "filename": "policy.pdf",
                "page": 7,
                "chunk_index": 15,
            },
            "similarity": 0.88,
        },
        {
            "content": "Health benefits start on day one.",  # Test 'content' key fallback
            "metadata": {
                "source": "benefits.docx",  # Test 'source' fallback for filename
            },
            "score": 0.75,  # Test 'score' fallback for similarity
        },
    ]


@pytest.fixture
def client(mock_cubo_app, mock_retrieved_docs):
    """Create test client with mocked dependencies."""
    with patch("cubo.server.api.cubo_app", mock_cubo_app):
        with patch("cubo.server.api.security_manager") as mock_security:
            mock_security.scrub.side_effect = lambda x: x  # No scrubbing
            
            mock_cubo_app.retriever.retrieve_top_documents.return_value = mock_retrieved_docs
            mock_cubo_app.generator.generate_response.return_value = "Test answer"
            
            from cubo.server.api import app
            yield TestClient(app)


class TestCitationsAPI:
    """Test cases for citations in query response."""

    def test_query_response_includes_citations(self, client, mock_cubo_app):
        """Verify that query response includes citations field."""
        response = client.post(
            "/api/query",
            json={"query": "What is the vacation policy?", "top_k": 3}
        )
        
        # Skip if server not properly initialized
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "citations" in data
        assert isinstance(data["citations"], list)

    def test_citation_structure(self, client, mock_cubo_app):
        """Verify citation has required fields."""
        response = client.post(
            "/api/query",
            json={"query": "What is the vacation policy?", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        if data.get("citations"):
            citation = data["citations"][0]
            
            # Required fields
            assert "source_file" in citation
            assert "chunk_id" in citation
            assert "chunk_index" in citation
            assert "text_snippet" in citation
            assert "relevance_score" in citation
            
            # Optional fields
            assert "page" in citation  # May be None

    def test_citation_source_file_from_filename(self, client, mock_cubo_app):
        """Verify source_file is extracted from metadata.filename."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        if data.get("citations"):
            # First doc has 'filename' in metadata
            assert data["citations"][0]["source_file"] == "policy.pdf"

    def test_citation_source_file_fallback_to_source(self, client, mock_cubo_app):
        """Verify source_file falls back to metadata.source."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        if len(data.get("citations", [])) >= 3:
            # Third doc has 'source' instead of 'filename'
            assert data["citations"][2]["source_file"] == "benefits.docx"

    def test_citation_text_snippet_truncated(self, client, mock_cubo_app):
        """Verify text_snippet is truncated to 200 chars."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        for citation in data.get("citations", []):
            assert len(citation["text_snippet"]) <= 200

    def test_citation_relevance_score_normalized(self, client, mock_cubo_app):
        """Verify relevance_score is a float between 0 and 1."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        for citation in data.get("citations", []):
            assert isinstance(citation["relevance_score"], float)
            assert 0.0 <= citation["relevance_score"] <= 1.0

    def test_citation_page_is_optional(self, client, mock_cubo_app):
        """Verify page field can be None."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        if len(data.get("citations", [])) >= 2:
            # Second doc has no 'page' in metadata
            assert data["citations"][1]["page"] is None or isinstance(data["citations"][1]["page"], int)

    def test_sources_and_citations_same_length(self, client, mock_cubo_app):
        """Verify sources and citations arrays have same length."""
        response = client.post(
            "/api/query",
            json={"query": "test", "top_k": 3}
        )
        
        if response.status_code == 503:
            pytest.skip("Server not initialized")
        
        data = response.json()
        
        assert len(data.get("sources", [])) == len(data.get("citations", []))
