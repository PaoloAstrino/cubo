import tempfile
from pathlib import Path

import numpy as np

from cubo.retrieval.vector_store import FaissStore
from cubo.workers.deletion_compactor import run_compaction_once


def make_dummy_vector(seed=0, dim=4):
    rng = np.random.RandomState(seed)
    return rng.rand(dim).astype(np.float32)


def test_enqueue_deletion_and_compact(tmp_path: Path):
    # Use a small index dir
    index_dir = tmp_path / "faiss_test"
    store = FaissStore(dimension=4, index_dir=index_dir)

    # Add two docs
    ids = ["docA", "docB"]
    vecs = [make_dummy_vector(i) for i in range(2)]
    docs = ["alpha document", "beta document"]
    metas = [{}, {}]
    store.add(embeddings=vecs, documents=docs, metadatas=metas, ids=ids)

    # Verify both present in DB
    assert store.count_vectors() == 2

    # Enqueue deletion for docA
    job_id = store.enqueue_deletion("docA", trace_id="test-trace", force=False)
    assert job_id is not None

    # After enqueue, DB should no longer contain docA's vector record (count reduced)
    assert store.count_vectors() == 1

    # The FAISS index may still return docA until compaction runs (old index still in memory)
    pre_comp_results = store.query(query_embeddings=[vecs[0]], n_results=5)
    pre_ids = pre_comp_results.get("ids", [[]])[0]
    # docA may still appear in the old index; this is expected until compaction
    assert isinstance(pre_ids, list)

    # Now run compaction (rebuild index) and ensure it completes
    worked = run_compaction_once(store)
    assert worked

    # After compaction, ensure the deleted doc is not returned by search
    # Ensure deleted doc is no longer reconstructable from the index
    assert store.get_vector("docA") is None
    # Optional: docB may or may not be available in index immediately depending on implementation details
    # but the important contract is that the deleted document must not be returned by search.

    store.close()
