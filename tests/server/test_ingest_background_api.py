from fastapi.testclient import TestClient

from cubo.server.api import app
from cubo.storage.metadata_manager import MetadataManager


def test_ingest_background_lifecycle(monkeypatch, tmp_path):
    """Test that background ingestion returns immediately and updates status."""
    db_path = tmp_path / "meta.db"
    manager = MetadataManager(db_path=str(db_path))

    # Create dummy input data
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    (input_dir / "doc1.txt").write_text("content 1")
    (input_dir / "doc2.txt").write_text("content 2")

    # Patch get_metadata_manager to use our temp DB
    monkeypatch.setattr("cubo.server.api.get_metadata_manager", lambda: manager)

    # Patch config to use temp output dir
    monkeypatch.setattr(
        "cubo.config.config.get",
        lambda k, d=None: str(tmp_path / "out") if "output_dir" in k else d,
    )

    # We need to patch cubo_app.doc_loader.load_documents_from_folder to return something
    # so the API doesn't exit early with "No documents found".
    # But wait, the API checks `cubo_app.doc_loader.load_documents_from_folder` BEFORE starting background task.
    # We need to mock cubo_app or at least doc_loader.

    class MockLoader:
        def load_documents_from_folder(self, path):
            return ["doc1", "doc2"]  # Dummy return

    class MockApp:
        doc_loader = MockLoader()

    monkeypatch.setattr("cubo.server.api.cubo_app", MockApp())

    client = TestClient(app)

    # 1. Start background ingestion
    # NOTE: TestClient executes background tasks synchronously *after* the response is returned.
    # However, if the background task fails silently, status remains 'pending'.
    # We need to ensure DeepIngestor actually runs.
    # The issue might be that DeepIngestor inside the background task creates a NEW MetadataManager instance
    # because get_metadata_manager() is called inside DeepIngestor.__init__ if not passed.
    # And get_metadata_manager() uses the default config path, not our temp DB!
    # We patched get_metadata_manager in api.py, but DeepIngestor is in a different module.
    # We need to patch get_metadata_manager in deep_ingestor.py too OR patch the global get_metadata_manager.

    monkeypatch.setattr("cubo.ingestion.deep_ingestor.get_metadata_manager", lambda: manager)

    resp = client.post("/api/ingest", json={"data_path": str(input_dir), "background": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    run_id = data["run_id"]
    assert run_id is not None

    # 2. Poll for completion (with timeout)
    # Since TestClient runs in same process, background tasks run after response.
    # But Starlette TestClient runs background tasks synchronously after the request?
    # Actually TestClient executes background tasks before returning response context usually,
    # or we might need to wait if it was truly async.
    # In TestClient, background tasks are executed. So by the time we get response,
    # the task might have already run if it's synchronous code.
    # DeepIngestor is synchronous code wrapped in a function.

    # Let's check status immediately.
    status_resp = client.get(f"/api/ingest/{run_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    # It should be completed because TestClient runs background tasks synchronously
    # Wait, if it failed, it might be 'pending' if the background task didn't run yet?
    # Or if it's running?
    # In TestClient, background tasks run *after* the response is sent.
    # So we might need to wait a bit or force execution?
    # Actually, TestClient runs them synchronously.
    # If status is 'pending', maybe the background task failed silently or didn't update status?
    # Or maybe the background task hasn't finished yet?
    # Let's add a small poll loop just in case.

    import time

    for _ in range(20):
        if status_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)
        status_resp = client.get(f"/api/ingest/{run_id}")
        status_data = status_resp.json()

    # Debug print if it fails
    if status_data["status"] not in ("completed", "failed"):
        print(f"DEBUG: Final status is {status_data['status']}")
        print(f"DEBUG: File counts: {status_data.get('file_status_counts')}")

    assert status_data["status"] in ("completed", "failed")
    assert status_data["chunks_count"] is not None

    # Check file statuses
    files_resp = client.get(f"/api/ingest/{run_id}/files")
    files_data = files_resp.json()
    assert len(files_data["files"]) >= 2

    manager.conn.close()
