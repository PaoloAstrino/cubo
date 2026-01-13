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


def test_query_with_empty_collection(client: TestClient, cubo_app, mini_data):
    """Test that querying with an existing but empty collection returns 400."""
    from cubo.server import api as server_api

    # Use lightweight cubo_app fixture for server
    server_api.cubo_app = cubo_app

    # Ingest mini dataset and build index so global vector store is non-empty
    r = client.post("/api/ingest", json={"data_path": mini_data, "fast_pass": True})
    assert r.status_code == 200

    r2 = client.post("/api/build-index", json={"force_rebuild": True})
    assert r2.status_code == 200

    # Create a new collection via API (will be empty)
    res = client.post("/api/collections", json={"name": "empty-coll"})
    assert res.status_code == 200
    data = res.json()
    coll_id = data.get("collection_id") or data.get("id") or data.get("collection_id")

    # Sanity check: collection has zero documents
    coll = client.get(f"/api/collections/{coll_id}").json()
    assert coll.get("document_count", 0) == 0

    # Query with collection id should return 400
    response = client.post("/api/query", json={"query": "test", "collection_id": coll_id})

    assert response.status_code == 400
    assert "has no documents" in response.json().get("detail", "")
