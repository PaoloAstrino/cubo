import os
import tempfile

from cubo.storage.metadata_manager import MetadataManager


def _manager(tmp_dir: str) -> MetadataManager:
    db_path = os.path.join(tmp_dir, "metadata.db")
    return MetadataManager(db_path=db_path)


def test_ingestion_file_status_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        m = _manager(tmp)
        run_id = "run-1"
        path = "doc.pdf"

        m.mark_file_processing(run_id, path, size_bytes=123)
        status = m.get_file_status(run_id, path)
        assert status is not None
        assert status["status"] == "processing"
        assert status["size_bytes"] == 123
        assert status["attempts"] == 0

        m.mark_file_failed(run_id, path, "bad pdf")
        status = m.get_file_status(run_id, path)
        assert status["status"] == "failed"
        assert status["error"] == "bad pdf"
        assert status["attempts"] == 1

        m.mark_file_succeeded(run_id, path)
        status = m.get_file_status(run_id, path)
        assert status["status"] == "succeeded"
        assert status["attempts"] == 1  # attempts should not reset on success

        counts = m.get_file_status_counts(run_id)
        assert counts["succeeded"] == 1

        files = m.list_files_for_run(run_id)
        assert len(files) == 1
        assert files[0]["file_path"] == path
        m.conn.close()
