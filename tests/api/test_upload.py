"""Tests for upload endpoint."""
import pytest
from io import BytesIO
from fastapi.testclient import TestClient


def test_upload_file(client: TestClient, tmp_path):
    """Test file upload."""
    # Create test file
    test_content = b"This is a test document."
    test_file = BytesIO(test_content)
    
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", test_file, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["filename"] == "test.txt"
    assert data["size"] == len(test_content)
    assert "trace_id" in data
    assert "message" in data


def test_upload_file_without_file(client: TestClient):
    """Test upload without file."""
    response = client.post("/api/upload")
    
    assert response.status_code == 422  # Validation error


def test_upload_file_has_trace_id(client: TestClient):
    """Test upload returns trace_id in header."""
    test_file = BytesIO(b"test content")
    
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", test_file, "text/plain")}
    )
    
    assert response.status_code == 200
    assert "x-trace-id" in response.headers


def test_upload_pdf_file(client: TestClient):
    """Test PDF file upload."""
    test_file = BytesIO(b"%PDF-1.4 fake pdf content")
    
    response = client.post(
        "/api/upload",
        files={"file": ("document.pdf", test_file, "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "document.pdf"
