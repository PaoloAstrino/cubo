from unittest.mock import MagicMock

import pytest
pytest.importorskip("torch")

from sentence_transformers import SentenceTransformer

from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever


@pytest.fixture
def mock_model():
    model = MagicMock(spec=SentenceTransformer)
    model.get_sentence_embedding_dimension.return_value = 8
    return model


def test_hybrid_normalization_missing_auto_score_does_not_dominate(mock_model, tmp_path):
    """Regression: auto_merging missing similarity must not default to 1.0 and beat real results."""
    config.set("vector_store_path", str(tmp_path))
    config.set("collection_name", "test_hybrid_normalization_missing_auto_score")
    config.set("vector_store_backend", "memory")
    config.set("retrieval.score_normalization", "minmax")

    retriever = DocumentRetriever(mock_model, use_auto_merging=True, use_reranker=False)

    # Force an auto_merging_retriever that returns no similarity.
    retriever.auto_merging_retriever = MagicMock()
    retriever.auto_merging_retriever.retrieve.return_value = [
        {"document": "auto", "metadata": {"id": "a1"}}
    ]

    # Provide sentence results with a real score.
    retriever._retrieve_sentence_window = MagicMock(return_value=[
        {"document": "sent", "metadata": {"id": "s1"}, "similarity": 0.95},
    ])

    # Bypass dedup complexity for this test.
    retriever.orchestrator.deduplicate_results = MagicMock(side_effect=lambda xs, *_: xs)

    out = retriever._hybrid_retrieval("q", top_k=2, strategy=None, trace_id=None)

    assert len(out) >= 1
    assert out[0].get("metadata", {}).get("id") == "s1"
    # Auto candidate should have raw_similarity None and similarity normalized (0.0)
    auto = next((r for r in out if r.get("metadata", {}).get("id") == "a1"), None)
    assert auto is not None
    assert auto.get("raw_similarity") is None


def test_hybrid_normalization_adds_fields(mock_model, tmp_path):
    config.set("vector_store_path", str(tmp_path))
    config.set("collection_name", "test_hybrid_normalization_adds_fields")
    config.set("vector_store_backend", "memory")
    config.set("retrieval.score_normalization", "minmax")

    retriever = DocumentRetriever(mock_model, use_auto_merging=True, use_reranker=False)
    retriever.auto_merging_retriever = MagicMock()
    retriever.auto_merging_retriever.retrieve.return_value = [
        {"document": "auto", "metadata": {"id": "a1"}, "similarity": 100.0}
    ]
    retriever._retrieve_sentence_window = MagicMock(return_value=[
        {"document": "sent", "metadata": {"id": "s1"}, "similarity": 0.1},
    ])
    retriever.orchestrator.deduplicate_results = MagicMock(side_effect=lambda xs, *_: xs)

    out = retriever._hybrid_retrieval("q", top_k=2, strategy=None, trace_id=None)
    assert len(out) == 2

    for r in out:
        assert "source" in r
        assert "raw_similarity" in r
        assert "normalized_similarity" in r
        # similarity should be the normalized score in [0,1]
        assert 0.0 <= float(r.get("similarity", 0.0)) <= 1.0
