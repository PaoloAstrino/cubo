from pathlib import Path
import time
import pytest
from src.cubo.indexing.index_publisher import cleanup


def test_cleanup_retention(tmp_path: Path):
    root = tmp_path / 'idxroot'
    root.mkdir(parents=True)
    # Create 4 versioned dirs
    for i in range(4):
        d = root / f'faiss_v{i+1}'
        d.mkdir()
        # create a dummy file
        (d / 'metadata.json').write_text('{}')
        # Wait a bit to vary mtime
        time.sleep(0.01)

    # Clean to keep last 2
    cleanup(root, keep_last_n=2)
    dirs = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('faiss_v')]
    assert len(dirs) == 2