import pytest

pytest.importorskip("fastapi")
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app():
    app = MagicMock()
    app.vector_store = MagicMock()
    app.retriever = MagicMock()
    # Simulate documents cache on app.state
    class DummyCache:
        def __init__(self, docs):
            self._docs = docs
        async def get(self):
            # return (list_of_docs, etag)
            return ([{"name": d} for d in self._docs], "etag-1")
        def invalidate(self):
            pass

    from types import SimpleNamespace
    mock_state = SimpleNamespace(documents_cache=DummyCache(["a.pdf", "b.docx"]))
    app.state = mock_state
    return app


@pytest.fixture
def client(mock_cubo_app):
    with patch("cubo.server.api.cubo_app", mock_cubo_app):
        from cubo.server.api import app

        yield TestClient(app)


class TestBulkDeleteAPI:
    def test_bulk_delete_enqueues_jobs(self, client, mock_cubo_app):
        mock_cubo_app.vector_store.enqueue_deletion.side_effect = lambda doc_id, **kw: f"job-{doc_id}"

        response = client.delete("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2
        assert len(data["queued"]) == 2
        assert data["queued"][0]["job_id"] == "job-a.pdf"

    def test_bulk_delete_force_true(self, client, mock_cubo_app):
        mock_cubo_app.vector_store.enqueue_deletion.side_effect = lambda doc_id, **kw: f"job-{doc_id}"

        response = client.delete("/api/documents?force=true")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2

    def test_bulk_delete_no_app(self, client):
        with patch("cubo.server.api.cubo_app", None):
            response = client.delete("/api/documents")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"].lower()