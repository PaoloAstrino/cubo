from fastapi.testclient import TestClient

from cubo.server.api import app
from cubo.storage.metadata_manager import MetadataManager


def test_ingest_status_endpoints(monkeypatch, tmp_path):
    db_path = tmp_path / "meta.db"
    manager = MetadataManager(db_path=str(db_path))

    run_id = "run-test"
    manager.record_ingestion_run(run_id, "./data", 0, None)
    manager.mark_file_processing(run_id, "file1.pdf")
    manager.mark_file_succeeded(run_id, "file1.pdf")
    manager.mark_file_failed(run_id, "file2.pdf", "encrypted")

    # Patch API to use the temp metadata manager
    monkeypatch.setattr("cubo.server.api.get_metadata_manager", lambda: manager)

    client = TestClient(app)

    run_resp = client.get(f"/api/ingest/{run_id}")
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert data["run_id"] == run_id
    assert data["file_status_counts"].get("succeeded") == 1
    assert data["file_status_counts"].get("failed") == 1

    files_resp = client.get(f"/api/ingest/{run_id}/files")
    assert files_resp.status_code == 200
    files_data = files_resp.json()
    assert files_data["run_id"] == run_id
    assert len(files_data["files"]) == 2

    manager.conn.close()
