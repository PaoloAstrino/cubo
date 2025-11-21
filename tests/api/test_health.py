"""Tests for health endpoint."""
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "version" in data
    assert "components" in data
    assert data["version"] == "1.0.0"


def test_health_check_has_trace_id(client: TestClient):
    """Test health check returns trace_id."""
    response = client.get("/api/health")
    
    assert response.status_code == 200
    assert "x-trace-id" in response.headers


def test_health_check_custom_trace_id(client: TestClient):
    """Test health check respects custom trace_id."""
    custom_trace_id = "test-trace-123"
    
    response = client.get(
        "/api/health",
        headers={"x-trace-id": custom_trace_id}
    )
    
    assert response.status_code == 200
    assert response.headers["x-trace-id"] == custom_trace_id
