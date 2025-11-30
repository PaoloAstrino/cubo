import threading
import time
from pathlib import Path

from cubo.indexing.faiss_index import FAISSIndexManager


def _fake_embeddings(count: int, dimension: int = 2):
    # Generate deterministic embeddings of the requested dimension so tests can
    # validate behavior with varying dimensions.
    return [[float(i + j) for j in range(dimension)] for i in range(count)]


def test_faiss_index_build_search_and_persist(tmp_path: Path):
    embeddings = _fake_embeddings(20)
    ids = [f"id_{i}" for i in range(len(embeddings))]
    index_dir = tmp_path / "faiss"
    manager = FAISSIndexManager(
        dimension=2, index_dir=index_dir, nlist=4, m=2, hnsw_m=8, hot_fraction=0.5
    )
    manager.build_indexes(embeddings, ids)
    assert len(manager.hot_ids) == 10
    assert len(manager.cold_ids) == 10
    assert manager.hot_index is not None
    assert manager.cold_index is not None

    hits = manager.search([0.1, 1.1], k=3)
    assert hits
    assert hits[0]["id"] in manager.hot_ids + manager.cold_ids

    manager.save()
    reloaded = FAISSIndexManager(dimension=2, index_dir=index_dir)
    reloaded.load()
    reloaded_hits = reloaded.search([0.1, 1.1], k=2)
    assert reloaded_hits


def test_faiss_with_opq(tmp_path: Path):
    """Test FAISS index building with OPQ (Optimized Product Quantization)."""
    # Use a sufficiently large number of embeddings so PQ/OPQ training has
    # enough samples to build codebooks (nbits=8 => 256 centroids per subquantizer).
    embeddings = _fake_embeddings(512, dimension=64)
    ids = [f"id_{i}" for i in range(len(embeddings))]
    index_dir = tmp_path / "faiss_opq"

    # Build index with OPQ enabled
    manager = FAISSIndexManager(
        dimension=64,
        index_dir=index_dir,
        nlist=4,
        m=8,
        hnsw_m=8,
        hot_fraction=0.3,
        use_opq=True,
        opq_m=16,
    )
    manager.build_indexes(embeddings, ids)

    # Verify indexes were built
    assert manager.hot_index is not None
    assert manager.cold_index is not None

    # Test search with OPQ
    query = [float(i) for i in range(64)]
    hits = manager.search(query, k=5)
    assert len(hits) > 0
    assert all(h["id"] in ids for h in hits)

    # Save and reload to ensure OPQ config persists
    manager.save()
    reloaded = FAISSIndexManager(dimension=64, index_dir=index_dir)
    reloaded.load()

    # Verify OPQ config was loaded
    assert reloaded.use_opq == True
    assert reloaded.opq_m == 16

    # Test search on reloaded index
    reloaded_hits = reloaded.search(query, k=5)
    assert len(reloaded_hits) > 0


def test_swap_indexes_basic(tmp_path: Path):
    """Test basic swap_indexes flow: build, save, load, swap."""
    # Build and save initial index
    embeddings_v1 = _fake_embeddings(6)
    ids_v1 = [f"v1_id_{i}" for i in range(len(embeddings_v1))]
    dir_v1 = tmp_path / "index_v1"
    manager = FAISSIndexManager(dimension=2, index_dir=dir_v1, nlist=2, hnsw_m=8, hot_fraction=0.5)
    manager.build_indexes(embeddings_v1, ids_v1)
    manager.save()

    # Verify initial search works
    hits_before = manager.search([0.1, 1.1], k=2)
    assert all(h["id"] in ids_v1 for h in hits_before)

    # Build new index in separate directory
    embeddings_v2 = _fake_embeddings(8)
    ids_v2 = [f"v2_id_{i}" for i in range(len(embeddings_v2))]
    dir_v2 = tmp_path / "index_v2"
    manager_v2 = FAISSIndexManager(
        dimension=2, index_dir=dir_v2, nlist=2, hnsw_m=8, hot_fraction=0.5
    )
    manager_v2.build_indexes(embeddings_v2, ids_v2)
    manager_v2.save()

    # Swap to new index
    manager.swap_indexes(dir_v2)

    # Verify new index is active
    hits_after = manager.search([0.1, 1.1], k=2)
    assert all(h["id"] in ids_v2 for h in hits_after)
    assert manager.index_dir == dir_v2


def test_atomic_swap(tmp_path: Path):
    # 1. Build initial index
    initial_embeddings = _fake_embeddings(10)
    initial_ids = [f"id_{i}" for i in range(len(initial_embeddings))]
    index_dir = tmp_path / "faiss_v1"
    manager = FAISSIndexManager(dimension=2, index_dir=index_dir)
    manager.build_indexes(initial_embeddings, initial_ids)
    manager.save()

    # 2. Start a search thread
    search_results = []
    stop_event = threading.Event()

    def search_thread():
        while not stop_event.is_set():
            hits = manager.search([0.1, 1.1], k=1)
            if hits:
                search_results.append(hits[0]["id"])
            time.sleep(0.01)

    thread = threading.Thread(target=search_thread)
    thread.start()

    # Let the search thread run for a bit
    time.sleep(0.1)

    # 3. Build a new index
    new_embeddings = _fake_embeddings(20)
    new_ids = [f"id_{i}" for i in range(len(new_embeddings))]
    new_index_dir = tmp_path / "faiss_v2"
    new_manager = FAISSIndexManager(dimension=2, index_dir=new_index_dir)
    new_manager.build_indexes(new_embeddings, new_ids)
    new_manager.save()

    # 4. Swap the indexes
    manager.swap_indexes(new_index_dir)

    # Let the search thread run for a bit more
    time.sleep(0.1)

    # 5. Stop the search thread
    stop_event.set()
    thread.join()

    # 6. Assert the results
    # The search thread should have started by getting results from the old index
    # and finished by getting results from the new index.
    assert any(id in initial_ids for id in search_results)
    assert any(id in new_ids for id in search_results)
    # The last result should be from the new index
    assert search_results[-1] in new_ids
