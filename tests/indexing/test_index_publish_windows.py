from pathlib import Path

import pytest

from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.indexing.index_publisher import get_current_index_dir, publish_version
from src.cubo.storage.metadata_manager import get_metadata_manager


def _fake_embeddings(count: int, dimension: int = 2):
    return [[float(i + j) for j in range(dimension)] for i in range(count)]


def test_publish_and_pointer_flip(tmp_path: Path):
    # Build an index v1 and publish
    index_root = tmp_path / "indexes"
    v1 = index_root / "faiss_v1"
    manager = FAISSIndexManager(dimension=2, index_dir=v1)
    embs = _fake_embeddings(10)
    ids = [f"id_{i}" for i in range(len(embs))]
    manager.build_indexes(embs, ids)
    manager.save(path=v1)

    # Publishing v1 should create pointer and metadata DB entry
    published = publish_version(v1, index_root)
    assert published == v1
    ptr = get_current_index_dir(index_root)
    assert ptr == v1

    # Now create a partially written v2 to simulate failure
    v2 = index_root / "faiss_v2"
    v2.mkdir(parents=True)
    # Create metadata only but no index files (simulate a failure scenario)
    with open(v2 / "metadata.json", "w", encoding="utf-8") as fh:
        fh.write("{}")

    # Attempt publish with verify=True should raise
    with pytest.raises(Exception):
        publish_version(v2, index_root, verify=True)

    # Pointer should remain unchanged
    assert get_current_index_dir(index_root) == v1

    # Create a valid v3 and publish
    v3 = index_root / "faiss_v3"
    manager3 = FAISSIndexManager(dimension=2, index_dir=v3)
    embs3 = _fake_embeddings(12)
    ids3 = [f"id_{i}" for i in range(len(embs3))]
    manager3.build_indexes(embs3, ids3)
    manager3.save(path=v3)
    published3 = publish_version(v3, index_root)
    assert get_current_index_dir(index_root) == v3

    # DB should have the latest version recorded
    latest = get_metadata_manager().get_latest_index_version()
    assert latest is not None
    assert latest["index_dir"] == str(v3)
