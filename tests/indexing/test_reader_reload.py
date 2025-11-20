from pathlib import Path
import pytest
pytest.importorskip("faiss")
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.indexing.index_publisher import publish_version, get_current_index_dir


def create_and_publish(index_root: Path, vname: str, vectors, ids):
    vdir = index_root / vname
    manager = FAISSIndexManager(dimension=len(vectors[0]), index_dir=vdir)
    manager.build_indexes(vectors, ids)
    manager.save(path=vdir)
    publish_version(vdir, index_root)
    return vdir


def test_reader_reload(tmp_path: Path):
    index_root = tmp_path / 'indexes'
    index_root.mkdir(parents=True)

    # Publish v1
    v1 = create_and_publish(index_root, 'faiss_v1', [[1.0, 0.0], [0.0, 1.0]], ['v1_a', 'v1_b'])

    # Reader loads current index via pointer
    reader = FAISSIndexManager(dimension=2, index_root=index_root)
    reader.load()
    ptr = get_current_index_dir(index_root)
    assert ptr == v1
    res1 = reader.search([1.0, 0.0], k=1)
    assert res1 and res1[0]['id'].startswith('v1_')

    # Publish v2
    v2 = create_and_publish(index_root, 'faiss_v2', [[10.0, 0.0], [0.0, 10.0]], ['v2_a', 'v2_b'])
    ptr = get_current_index_dir(index_root)
    assert ptr == v2

    # Reader explicitly swap to new index
    reader.swap_indexes(v2)
    res2 = reader.search([10.0, 0.0], k=1)
    assert res2 and res2[0]['id'].startswith('v2_')
