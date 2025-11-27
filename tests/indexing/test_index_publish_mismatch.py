from pathlib import Path

import pytest

from src.cubo.indexing.index_publisher import get_current_index_dir, publish_version


def test_publish_metadata_hot_ids_claim_but_missing_index(tmp_path: Path):
    root = tmp_path / "indexes"
    v = root / "faiss_v_bad"
    v.mkdir(parents=True)
    # Create metadata speccing a dimension and hot_ids but no hot.index
    with open(v / "metadata.json", "w", encoding="utf-8") as fh:
        fh.write('{"dimension":2, "hot_ids": ["id_1"]}')

    with pytest.raises(Exception):
        publish_version(v, root, verify=True)

    assert get_current_index_dir(root) is None
