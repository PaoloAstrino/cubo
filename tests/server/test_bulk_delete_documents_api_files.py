import pytest

pytest.importorskip("fastapi")
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def mock_cubo_app_with_files():
    app = MagicMock()
    app.vector_store = MagicMock()
    app.retriever = MagicMock()

    # Simulate documents cache on app.state with files
    class DummyCache:
        def __init__(self, docs):
            self._docs = docs

        async def get(self):
            return ([{"name": d} for d in self._docs], "etag-1")

        def invalidate(self):
            pass

    from types import SimpleNamespace

    mock_state = SimpleNamespace(documents_cache=DummyCache(["a.pdf", "b.docx"]))
    app.state = mock_state
    return app


@pytest.fixture
def client(mock_cubo_app_with_files):
    with patch("cubo.server.api.cubo_app", mock_cubo_app_with_files):
        from cubo.server.api import app

        yield TestClient(app)


class TestBulkDeleteFiles:
    def test_bulk_delete_removes_files(self, client, mock_cubo_app_with_files, tmp_path):
        # Create files in data dir
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "a.pdf").write_text("a")
        (data_dir / "b.docx").write_text("b")
        assert (data_dir / "a.pdf").exists()
        assert (data_dir / "b.docx").exists()

        mock_cubo_app_with_files.vector_store.enqueue_deletion.side_effect = (
            lambda doc_id, **kw: f"job-{doc_id}"
        )

        response = client.delete("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 2
        assert len(data["queued"]) == 2

        # Files should be removed from disk
        assert not (data_dir / "a.pdf").exists()
        assert not (data_dir / "b.docx").exists()
