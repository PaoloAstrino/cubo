import numpy as np
import tempfile
from pathlib import Path

from cubo.retrieval.scaffold_retriever import SimpleIndex
from cubo.processing.scaffold import ScaffoldGenerator


def test_simpleindex_accepts_python_list_and_searches():
    # 3 embeddings of dim 4
    emb_list = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    idx = SimpleIndex(emb_list)

    # query similar to first vector
    q = np.array([1.0, 0.0, 0.0, 0.0])
    dists, inds = idx.search(q, k=2)

    assert dists.shape == (1, 2)
    assert inds.shape == (1, 2)
    # best match should be index 0
    assert int(inds[0, 0]) == 0

    # Also accept Python lists as query (defensive)
    dists2, inds2 = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert dists2.shape == (1, 2)
    assert int(inds2[0, 0]) == 0


def test_load_scaffolds_returns_ndarray(tmp_path: Path):
    # create minimal scaffold files expected by load_scaffolds
    sg = ScaffoldGenerator.__new__(ScaffoldGenerator)

    # create fake parquet and mapping files (write a minimal dataframe)
    import pandas as pd

    pd.DataFrame([{"scaffold_id": "s1", "summary": "x"}]).to_parquet(tmp_path / "scaffold_metadata.parquet")
    mapping = {"s1": ["c1"]}
    (tmp_path / "scaffold_mapping.json").write_text("{\"s1\": [\"c1\"]}")

    # write a numpy embeddings file
    arr = np.random.randn(2, 8).astype(np.float32)
    np.save(tmp_path / "scaffold_embeddings.npy", arr)

    result = sg.load_scaffolds(tmp_path)
    assert hasattr(result["scaffold_embeddings"], "shape")
    assert result["scaffold_embeddings"].shape[1] == 8
