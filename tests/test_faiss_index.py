from pathlib import Path

from src.cubo.indexing.faiss_index import FAISSIndexManager


def _fake_embeddings(count: int, dimension: int = 2):
    return [[float(i), float(i + 1)] for i in range(count)]


def test_faiss_index_build_search_and_persist(tmp_path: Path):
    embeddings = _fake_embeddings(6)
    ids = [f"id_{i}" for i in range(len(embeddings))]
    index_dir = tmp_path / "faiss"
    manager = FAISSIndexManager(
        dimension=2,
        index_dir=index_dir,
        nlist=2,
        hnsw_m=8,
        hot_fraction=0.5
    )
    manager.build_indexes(embeddings, ids)
    assert len(manager.hot_ids) > 0
    assert manager.hot_index is not None

    hits = manager.search([0.1, 1.1], k=3)
    assert hits
    assert hits[0]['id'] in manager.hot_ids + manager.cold_ids

    manager.save()
    reloaded = FAISSIndexManager(dimension=2, index_dir=index_dir)
    reloaded.load()
    reloaded_hits = reloaded.search([0.1, 1.1], k=2)
    assert reloaded_hits