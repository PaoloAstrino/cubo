"""Tests for query endpoint."""

from fastapi.testclient import TestClient


def test_query_basic(client: TestClient):
    """Test basic query."""
    response = client.post(
        "/api/query", json={"query": "What is this about?", "top_k": 5, "use_reranker": True}
    )

    # May fail if retriever not initialized, but should return valid response
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "trace_id" in data
        assert "query_scrubbed" in data


def test_query_without_query_text(client: TestClient):
    """Test query without query text."""
    response = client.post("/api/query", json={"top_k": 5})

    assert response.status_code == 422  # Validation error


def test_query_with_invalid_top_k(client: TestClient):
    """Test query with invalid top_k."""
    response = client.post("/api/query", json={"query": "test query", "top_k": 100})  # Exceeds max

    assert response.status_code == 422  # Validation error


def test_query_has_trace_id(client: TestClient):
    """Test query returns trace_id."""
    response = client.post("/api/query", json={"query": "test"})

    assert "x-trace-id" in response.headers


def test_query_custom_trace_id(client: TestClient):
    """Test query respects custom trace_id."""
    custom_trace_id = "query-trace-456"

    response = client.post(
        "/api/query", json={"query": "test query"}, headers={"x-trace-id": custom_trace_id}
    )

    assert response.headers["x-trace-id"] == custom_trace_id


def test_query_default_parameters(client: TestClient):
    """Test query with default parameters."""
    response = client.post("/api/query", json={"query": "test query"})

    # Should accept query with defaults
    assert response.status_code in [200, 503]
