import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch
from src.cubo.indexing.index_publisher import publish_version
from src.cubo.indexing.index_publisher import POINTER_FILENAME


def create_dummy_index(tmp_path: Path):
    index_dir = tmp_path / 'faiss_vdummy'
    index_dir.mkdir(parents=True)
    # minimal metadata
    meta = {"dimension": 2, "hot_ids": [], "cold_ids": []}
    with open(index_dir / 'metadata.json', 'w', encoding='utf-8') as fh:
        json.dump(meta, fh)
    return index_dir


def test_publish_retries_on_replace(tmp_path):
    # Prepare a fake index root
    index_root = tmp_path / 'indexes'
    index_root.mkdir()
    version_dir = create_dummy_index(tmp_path)

    # Prepare a replacer sequence that raises PermissionError first two times
    calls = {"count": 0}

    def fake_replace(src, dst):
        calls['count'] += 1
        if calls['count'] < 3:
            raise PermissionError(13, 'Access is denied')
        # Simulate a successful replace by making a copy
        with open(src, 'r', encoding='utf-8') as fh_src:
            with open(dst, 'w', encoding='utf-8') as fh_dst:
                fh_dst.write(fh_src.read())

    with patch('src.cubo.indexing.index_publisher.os.replace', new=fake_replace):
        # Should not raise
        published = publish_version(version_dir, index_root, verify=False)
        assert published == version_dir
        pointer = index_root / POINTER_FILENAME
        assert pointer.exists()

    assert calls['count'] >= 3
