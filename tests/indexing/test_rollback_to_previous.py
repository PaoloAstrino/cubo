from pathlib import Path

import pytest

from tests.utils import create_and_publish_faiss_index

pytest.importorskip("faiss")
pytestmark = pytest.mark.requires_faiss


def test_rollback_to_previous(tmp_path: Path, tmp_metadata_db):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    # Monkeypatch index_publisher to use our temp metadata manager before any publish
    from src.cubo.indexing import index_publisher as ip
    from src.cubo.storage import metadata_manager as mm
    from src.cubo.storage.metadata_manager import MetadataManager

    monkey_manager = MetadataManager(db_path=tmp_metadata_db)
    ip.get_metadata_manager = lambda: monkey_manager
    # Patch both module-level manager and publisher helper to use same manager
    mm._manager = monkey_manager
    manager = mm._manager
    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=8, dim=2)
    assert any("faiss_v1" in v["index_dir"] for v in manager.list_index_versions(limit=5))
    v2, _, _ = create_and_publish_faiss_index(index_root, "faiss_v2", n_vectors=8, dim=2)
    # Ensure DB explicitly records both versions to avoid racey behavior
    manager.record_index_version("faiss_v1", str(v1))
    manager.record_index_version("faiss_v2", str(v2))
    assert any("faiss_v2" in v["index_dir"] for v in manager.list_index_versions(limit=5))

    from src.cubo.indexing.index_publisher import get_current_index_dir, rollback_to_previous

    # Sanity: current pointer should be v2
    assert get_current_index_dir(index_root) == v2

    # Rollback via helper
    events = []

    def telemetry_hook(event, payload):
        events.append((event, payload))

    # use the same manager as above for consistency; the earlier assertions already verified DB content

    import json

    pointer_path = index_root / "current_index.json"
    versions_pre_rb = manager.list_index_versions(limit=5)
    ok = rollback_to_previous(index_root, telemetry_hook=telemetry_hook)
    versions_post_rb = manager.list_index_versions(limit=5)
    with open(pointer_path, encoding="utf-8") as pfh:
        _after = json.load(pfh)
    assert ok is True
    assert get_current_index_dir(index_root) == v1
    # Telemetry saw a rolled_back event
    assert any(e == "rolled_back" for (e, _p) in events)


def test_rollback_no_previous(tmp_path: Path, tmp_metadata_db):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    # Publish only v1
    # Monkeypatch publisher to use a per-test metadata manager
    from src.cubo.indexing import index_publisher as ip
    from src.cubo.storage import metadata_manager as mm
    from src.cubo.storage.metadata_manager import MetadataManager

    mm._manager = MetadataManager(db_path=tmp_metadata_db)
    ip.get_metadata_manager = lambda: mm._manager
    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=8, dim=2)

    from src.cubo.indexing.index_publisher import get_current_index_dir, rollback_to_previous

    assert get_current_index_dir(index_root) == v1
    ok = rollback_to_previous(index_root)
    assert ok is False
    assert get_current_index_dir(index_root) == v1
