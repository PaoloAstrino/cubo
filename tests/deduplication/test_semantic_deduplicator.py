import pytest
pytest.importorskip("torch")

import json

import numpy as np
import pandas as pd

from cubo.deduplication.semantic_deduplicator import HybridDeduplicator


def test_semantic_dedup_run_creates_canonical_map(tmp_path):
    df = pd.DataFrame(
        [
            {"chunk_id": "c1", "text": "alpha beta", "summary_score": 0.2},
            {"chunk_id": "c2", "text": "alpha beta", "summary_score": 0.9},
            {"chunk_id": "c3", "text": "gamma delta", "summary_score": 0.5},
        ]
    )
    embeddings = np.array(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    deduplicator = HybridDeduplicator(
        method="semantic",
        similarity_threshold=0.9,
        representative_metric="summary_score",
        prefilter={"use_minhash": False},
        ann={"backend": "sklearn", "k": 2},
        clustering={"algorithm": "graph"},
    )

    output_path = tmp_path / "dedup_map.json"
    result = deduplicator.run(df, embeddings, output_map_path=str(output_path))

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert len(payload["canonical_map"]) == 3
    # c1 and c2 should collapse into a single canonical id (c2 due to higher summary score)
    reps = {rep["chunk_id"] for rep in payload["representatives"].values()}
    assert "c2" in reps
    assert len(set(result.canonical_map.values())) == 2


def test_select_representatives_prefers_summary_score():
    df = pd.DataFrame(
        [
            {"chunk_id": "a", "text": "foo", "summary_score": 0.1},
            {"chunk_id": "b", "text": "foobar", "summary_score": 0.7},
        ]
    )
    deduplicator = HybridDeduplicator(representative_metric="summary_score")
    clusters = [{"a", "b"}]

    reps = deduplicator.select_representatives(clusters, df)

    assert reps[0]["chunk_id"] == "b"
    assert reps[0]["score"] == 0.7
