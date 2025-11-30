from pathlib import Path

import pytest

from tests.utils import create_and_publish_faiss_index

pytest.importorskip("faiss")
pytestmark = pytest.mark.requires_faiss


def test_publish_db_atomicity(tmp_path: Path, monkeypatch, tmp_metadata_db):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    # Publish v1
    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=8, dim=2)

    from cubo.indexing.index_publisher import get_current_index_dir, publish_version

    # Ensure v1 is the current pointer
    assert get_current_index_dir(index_root) == v1

    # Create v2 directory but simulate DB failure at record_index_version time
    v2 = index_root / "faiss_v2"
    v2.mkdir(parents=True)
    # Setup a valid index under v2
    from cubo.indexing.faiss_index import FAISSIndexManager

    manager = FAISSIndexManager(dimension=2, index_dir=v2)
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    ids = ["v2_a", "v2_b"]
    manager.build_indexes(vectors, ids)
    manager.save(path=v2)

    # Monkeypatch the metadata manager record method to raise
    class FakeManager:
        def record_index_version(self, version_id, index_dir):
            raise RuntimeError("Simulated DB failure during record")

        def get_latest_index_version(self):
            return {"id": "faiss_v1", "index_dir": str(v1)}

    from cubo.indexing import index_publisher as ip

    monkeypatch.setattr(ip, "get_metadata_manager", lambda: FakeManager())

    # Attempt publish and observe behavior: current desired behavior is to maintain pointer to previous
    with pytest.raises(RuntimeError):
        publish_version(v2, index_root)

    # Ensure pointer still points to v1 (desired rollback semantics)
    assert get_current_index_dir(index_root) == v1
