import concurrent.futures
import os
import tempfile

from cubo.storage.metadata_manager import MetadataManager


def _new_manager(tmp_dir: str) -> MetadataManager:
    db_path = os.path.join(tmp_dir, "metadata.db")
    return MetadataManager(db_path=db_path)


def test_concurrent_ingestion_runs_thread_safe():
    """Concurrent ingestion run inserts should not raise thread errors or lock the DB."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _new_manager(tmp)
        run_ids = [f"run-{i}" for i in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda rid: manager.record_ingestion_run(rid, "src", 1), run_ids))

        runs = manager.list_runs_by_status("pending")
        assert len(runs) == len(run_ids)
        manager.conn.close()


def test_concurrent_chunk_mappings_thread_safe():
    """Concurrent chunk mapping writes should succeed without sqlite locking."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = _new_manager(tmp)
        run_id = "run-chunks"
        payloads = [(run_id, f"old-{i}", f"new-{i}") for i in range(16)]

        def _write(item):
            r, old, new = item
            manager.add_chunk_mapping(r, old, new, {"idx": old})

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_write, payloads))

        mappings = manager.list_mappings_for_run(run_id)
        assert len(mappings) == len(payloads)
        manager.conn.close()
