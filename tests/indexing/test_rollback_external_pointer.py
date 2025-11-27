import json
from pathlib import Path

import pytest

from tests.utils import create_and_publish_faiss_index

pytest.importorskip("faiss")
pytestmark = pytest.mark.requires_faiss


def test_rollback_with_external_pointer(tmp_path: Path, tmp_metadata_db):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    from src.cubo.indexing import index_publisher as ip
    from src.cubo.storage import metadata_manager as mm
    from src.cubo.storage.metadata_manager import MetadataManager

    mm._manager = MetadataManager(db_path=tmp_metadata_db)
    ip.get_metadata_manager = lambda: mm._manager
    manager = mm._manager

    # Publish recorded v1 and v2
    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=8, dim=2)
    v2, _, _ = create_and_publish_faiss_index(index_root, "faiss_v2", n_vectors=8, dim=2)
    manager.record_index_version("faiss_v1", str(v1))
    manager.record_index_version("faiss_v2", str(v2))

    # Simulate external pointer update to v3 (not recorded in DB)
    v3 = index_root / "faiss_v3"
    v3.mkdir(parents=True)
    # Write a pointer JSON pointing to v3
    pointer_file = index_root / "current_index.json"
    payload_v3 = {"index_dir": str(v3), "timestamp": 123456, "version_id": "faiss_v3"}
    with open(pointer_file, "w", encoding="utf-8") as fh:
        json.dump(payload_v3, fh)

    from src.cubo.indexing.index_publisher import get_current_index_dir, rollback_to_previous

    ok = rollback_to_previous(index_root)
    assert ok is True
    assert get_current_index_dir(index_root) == v2
