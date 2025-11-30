import concurrent.futures
import time
from pathlib import Path

import pytest

from tests.utils import create_and_publish_faiss_index, reader_loop

pytest.importorskip("faiss")
pytestmark = pytest.mark.requires_faiss


def test_publish_reader_safety(tmp_path: Path, monkeypatch):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    # Publish v1
    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=16, dim=2)

    from cubo.indexing.faiss_index import FAISSIndexManager
    from cubo.indexing.index_publisher import get_current_index_dir

    reader = FAISSIndexManager(dimension=2, index_root=index_root)
    reader.load()

    # Run reader loop in background
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        future = ex.submit(reader_loop, reader, [0.0, 0.0], 200, 0.01)

        # Simulate slower save by monkeypatching faiss.write_index to sleep
        import faiss

        orig_write_index = faiss.write_index

        def slow_write_index(idx, path):
            time.sleep(0.05)
            return orig_write_index(idx, path)

        monkeypatch.setattr(faiss, "write_index", slow_write_index)

        # Publish v2 while readers are running
        v2, vectors, ids = create_and_publish_faiss_index(
            index_root, "faiss_v2", n_vectors=12, dim=2
        )

        # Wait for reader loop to finish
        results, errors = future.result()

        # Restore original write_index
        monkeypatch.setattr(faiss, "write_index", orig_write_index)

    # Ensure there were no exceptions from readers
    assert len(errors) == 0

    # Ensure pointer points to v2 now
    assert get_current_index_dir(index_root) == v2

    # Swap reader explicitly to new index
    reader.swap_indexes(v2)
    res = reader.search([float(vectors[0][0]), float(vectors[0][1])], k=1)
    assert res and res[0]["id"].startswith("id_")
