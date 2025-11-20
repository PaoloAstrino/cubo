import concurrent.futures
import time
from pathlib import Path
import pytest
from tests.utils import create_and_publish_faiss_index

pytest.importorskip('faiss')
pytestmark = pytest.mark.requires_faiss


def _reader_task(index_root: Path, iterations: int = 400, delay: float = 0.01):
    from src.cubo.indexing.faiss_index import FAISSIndexManager

    mgr = FAISSIndexManager(dimension=2, index_root=index_root)
    errors = []
    for _ in range(iterations):
        try:
            # Load will pick up the current pointer; searching uses a thread-safe lock
            mgr.load()
            _ = mgr.search([0.0, 0.0], k=1)
        except Exception as exc:
            errors.append(exc)
        time.sleep(delay)
    return errors


def _publisher_task(index_root: Path, count: int = 10, delay: float = 0.02):
    errors = []
    for i in range(1, count + 1):
        try:
            create_and_publish_faiss_index(index_root, f'faiss_v{i}', n_vectors=16, dim=2)
        except Exception as exc:
            errors.append(exc)
        time.sleep(delay)
    return errors


@pytest.mark.integration
@pytest.mark.e2e
def test_publish_stress_concurrency(tmp_path: Path, tmp_metadata_db):
    """A stress test that runs multiple concurrent readers while repeatedly publishing new indexes.
    The test asserts that publishers can create and publish versions without crashing and readers only
    encounter a low number of transient errors.
    """
    index_root = tmp_path / 'indexes'
    index_root.mkdir(parents=True)

    # Initial published version
    create_and_publish_faiss_index(index_root, 'faiss_v0', n_vectors=8, dim=2)

    num_readers = 4
    publish_count = 12

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_readers + 1) as ex:
        # Launch readers
        readers = [ex.submit(_reader_task, index_root, 600, 0.005) for _ in range(num_readers)]
        # Launch publisher
        publisher = ex.submit(_publisher_task, index_root, publish_count, 0.02)

        pub_errors = publisher.result(timeout=120)
        reader_errors = [r.result(timeout=120) for r in readers]

    # Publisher should not have errors
    assert len(pub_errors) == 0, f"Publisher had errors: {pub_errors}"

    # Allow a small number of transient reader errors due to pointer flips, but assert they are limited
    total_reader_errors = sum(len(e) for e in reader_errors)
    assert total_reader_errors < (num_readers * 10), f"Too many reader errors during stress test: {total_reader_errors}"

    # Ensure pointer points to the last published directory
    from src.cubo.indexing.index_publisher import get_current_index_dir
    last_published = get_current_index_dir(index_root)
    assert last_published is not None, "No pointer present after publishing"
    assert last_published.name == f'faiss_v{publish_count}', "Pointer did not end on the expected latest index"
