import time

from cubo.core import CuboCore


def test_async_ingestion_smoke(tmp_path):
    """Verify async ingestion returns immediately and completes in background."""
    core = CuboCore()

    # Setup dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.txt").write_text("Hello async world")

    # Mock initialize_components to avoid loading models
    core.initialize_components = lambda: True
    core.model = "mock"
    core.retriever = "mock"
    core.generator = "mock"

    # Mock build_index to simulate work
    def slow_build(folder=None):
        time.sleep(1)  # Simulate work
        return 5  # "processed 5 chunks"

    core.build_index = slow_build

    start = time.time()
    job_id = core.ingest_documents_async(str(data_dir))
    duration = time.time() - start

    # Assertion 1: Returns immediately (fast)
    assert duration < 0.1, "ingest_documents_async blocked main thread!"
    assert job_id is not None

    # Assertion 2: Status updates
    status = core.get_task_status(job_id)
    # Could be PENDING or RUNNING
    assert status["status"] in ["pending", "running"]

    # Wait for completion
    for _ in range(20):
        status = core.get_task_status(job_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(0.1)

    assert status["status"] == "completed"
    assert status["result"] == 5
