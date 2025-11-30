from pathlib import Path

import pytest

from tests.utils import create_and_publish_faiss_index

pytest.importorskip("faiss")
pytestmark = pytest.mark.requires_faiss


def test_publish_telemetry(tmp_path: Path):
    index_root = tmp_path / "indexes"
    index_root.mkdir(parents=True)

    v1, _, _ = create_and_publish_faiss_index(index_root, "faiss_v1", n_vectors=8, dim=2)
    v2, _, _ = create_and_publish_faiss_index(index_root, "faiss_v2", n_vectors=8, dim=2)

    from cubo.indexing.index_publisher import publish_version

    events = []

    def telemetry_hook(event, payload):
        events.append((event, payload))

    publish_version(v2, index_root, telemetry_hook=telemetry_hook)

    # Ensure telemetry saw db_recorded event
    assert any(evt == "db_recorded" for (evt, _p) in events)
