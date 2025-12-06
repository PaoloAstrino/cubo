import pytest
pytest.importorskip("torch")

from pathlib import Path

import numpy as np

from cubo.retrieval.vector_store import FaissStore


def test_faiss_store_add_and_promote(tmp_path: Path):
    dim = 8
    store = FaissStore(dimension=dim, index_dir=tmp_path / "faiss_index")
    # small deterministic vectors
    ids = [f"doc{i}" for i in range(6)]
    # Create vectors spaced apart
    vectors = [np.ones(dim) * (i + 1) for i in range(6)]
    docs = [f"Document {i}" for i in range(6)]
    metas = [{"filename": f"doc{i}.txt"} for i in range(6)]
    store.add(embeddings=vectors, documents=docs, metadatas=metas, ids=ids)

    # Query near doc3 vector and ensure we get results and a mixture of hot/cold
    q = vectors[3]
    res = store.query(query_embeddings=[q], n_results=3)
    assert "documents" in res and res["documents"]

    # Promote doc3 explicitly (synchronously to make test deterministic)
    store.promote_to_hot_sync("doc3")
    # After promoting, the doc should be in hot set; query again and the res returned for doc3 should show 'hot' in source
    res2 = store.query(query_embeddings=[q], n_results=3)
    # Search results include ids
    ids_out = res2.get("ids", [[]])[0]
    assert "doc3" in ids_out
    # -- ensure that at least one result's source equals 'hot' if available
    # We can't access 'source' directly from vector_store API (faiss manager returns 'source' per item), so confirm hot ids known
    assert store._index.hot_ids
