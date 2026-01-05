import pytest

pytest.importorskip("fastapi")
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app():
    app = MagicMock()
    app.vector_store = MagicMock()
    return app


@pytest.fixture
def client(mock_cubo_app):
    with patch("cubo.server.api.cubo_app", mock_cubo_app):
        from cubo.server.api import app

        yield TestClient(app)


def test_get_delete_status_found(client, mock_cubo_app):
    mock_cubo_app.vector_store.get_deletion_status.return_value = {
        "id": "job1",
        "doc_id": "docA",
        "status": "completed",
    }

    r = client.get("/api/delete-status/job1")
    assert r.status_code == 200
    assert r.json()["id"] == "job1"


def test_get_delete_status_not_found(client, mock_cubo_app):
    mock_cubo_app.vector_store.get_deletion_status.return_value = None
    r = client.get("/api/delete-status/unknown")
    assert r.status_code == 404
