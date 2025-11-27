import multiprocessing
from pathlib import Path

import pytest

pytest.importorskip("faiss")
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.indexing.index_publisher import get_current_index_dir, publish_version


def worker_publish(root: Path, vdir: Path, dim: int = 2):
    manager = FAISSIndexManager(dimension=dim, index_dir=vdir)
    embs = [[float(i + j) for j in range(dim)] for i in range(8)]
    ids = [f"id_{i}" for i in range(len(embs))]
    manager.build_indexes(embs, ids)
    manager.save(path=vdir)
    publish_version(vdir, root)


def test_concurrent_publish(tmp_path: Path):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    v1 = index_root / "faiss_v1"
    v2 = index_root / "faiss_v2"

    # Use multiprocessing to attempt to publish both concurrently
    p1 = multiprocessing.Process(target=worker_publish, args=(index_root, v1))
    p2 = multiprocessing.Process(target=worker_publish, args=(index_root, v2))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Pointer should exist and point to either v1 or v2
    ptr = get_current_index_dir(index_root)
    assert ptr in (v1, v2)

    # Both directories should exist and contain metadata.json
    assert (v1 / "metadata.json").exists()
    assert (v2 / "metadata.json").exists()
