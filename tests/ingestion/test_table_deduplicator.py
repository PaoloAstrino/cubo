import numpy as np
import pandas as pd
import pytest

from cubo.deduplication.table_deduplicator import TableDeduplicator


class FakeEmbedder:
    def encode(self, texts, convert_to_numpy=False):
        vectors = []
        for text in texts:
            length = len(text)
            commas = text.count(",")
            vectors.append([length, commas])
        arr = np.asarray(vectors, dtype=np.float32)
        return arr if convert_to_numpy else arr.tolist()


def _sample_table(chunk_id: str, columns, rows):
    return {
        "chunk_id": chunk_id,
        "metadata": {"columns": columns},
        "sample_rows": rows,
    }


def test_table_deduplicator_clusters_and_virtual_tables():
    tables = pd.DataFrame(
        [
            _sample_table("t1", ["id", "value"], [{"id": 1, "value": "x"}]),
            _sample_table("t2", ["id", "value"], [{"id": 2, "value": "y"}]),
            _sample_table("t3", ["user", "score"], [{"user": "a", "score": 5}]),
        ]
    )

    # Force sklearn fallback for deterministic clustering within unit tests (avoid HDBSCAN stochastic behavior)
    import cubo.deduplication.table_deduplicator as tdd

    tdd.hdbscan = None
    tdd.umap = None
    deduplicator = TableDeduplicator(FakeEmbedder(), min_cluster_size=2, min_samples=1)
    embeddings = deduplicator.embed_tables(tables)

    assert embeddings.shape[0] == len(tables)

    labels = deduplicator.cluster_tables(embeddings)
    virtual_tables = deduplicator.create_virtual_tables(tables, labels)

    assert isinstance(virtual_tables, list)
    assert any(
        vt["common_columns"] == ["id", "value"] for vt in virtual_tables if vt["source_tables"]
    )


def test_embed_tables_handles_empty_dataframe():
    tables = pd.DataFrame(columns=["chunk_id", "metadata", "sample_rows"])
    deduplicator = TableDeduplicator(FakeEmbedder())

    embeddings = deduplicator.embed_tables(tables)

    assert embeddings.size == 0

    labels = deduplicator.cluster_tables(embeddings)
    assert labels.size == 0


def test_create_virtual_tables_no_tables_returns_empty():
    tables = pd.DataFrame(columns=["chunk_id", "metadata", "sample_rows"])
    deduplicator = TableDeduplicator(FakeEmbedder())

    result = deduplicator.create_virtual_tables(tables, np.array([], dtype=int))

    assert result == []


def test_create_virtual_tables_requires_matching_label_lengths():
    tables = pd.DataFrame(
        [
            _sample_table("t1", ["id"], [{"id": 1}]),
            _sample_table("t2", ["id"], [{"id": 2}]),
        ]
    )
    deduplicator = TableDeduplicator(FakeEmbedder())

    with pytest.raises(ValueError):
        deduplicator.create_virtual_tables(tables, np.array([0], dtype=int))


def test_virtual_table_common_columns_with_mixed_schemas_and_noise():
    tables = pd.DataFrame(
        [
            _sample_table(
                "t1", ["id", "value", "timestamp"], [{"id": 1, "value": "x", "timestamp": "2020"}]
            ),
            _sample_table("t2", ["value", "id"], [{"id": 2, "value": "y"}]),
            _sample_table("t3", ["user", "score"], [{"user": "a", "score": 5}]),
            _sample_table(
                "t4", ["score", "user", "region"], [{"user": "b", "score": 3, "region": "na"}]
            ),
        ]
    )
    deduplicator = TableDeduplicator(FakeEmbedder(), min_cluster_size=1, min_samples=1)

    labels = np.array([0, 0, 1, -1], dtype=int)
    virtual_tables = deduplicator.create_virtual_tables(tables, labels)

    assert len(virtual_tables) == 2

    vt_by_cluster = {vt["cluster_id"]: vt for vt in virtual_tables}

    cluster_zero = vt_by_cluster[0]
    assert cluster_zero["common_columns"] == ["id", "value"]
    assert cluster_zero["source_tables"] == ["t1", "t2"]

    cluster_one = vt_by_cluster[1]
    assert cluster_one["common_columns"] == ["score", "user"]
    assert cluster_one["source_tables"] == ["t3"]
