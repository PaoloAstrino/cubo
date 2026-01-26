from unittest.mock import patch

import pytest

from cubo.rerank.reranker import CrossEncoderReranker


def test_crossencoder_reranker_scores_and_sorts():
    candidates = [
        {"document": "first doc", "content": "first doc content"},
        {"document": "second doc", "content": "second doc content"},
    ]

    # Patch CrossEncoder in sentence_transformers to simulate predictable scoring
    class FakeCrossEncoder:
        def __init__(self, name):
            self.name = name

        def predict(self, pairs):
            # return higher score for second candidate
            return [0.1, 0.9]

    with patch("sentence_transformers.CrossEncoder", FakeCrossEncoder):
        reranker = CrossEncoderReranker(model_name="fake-model", top_n=2)
        ranked = reranker.rerank("query", candidates, max_results=2)
        assert len(ranked) == 2
        assert ranked[0]["document"] == "second doc"
        assert ranked[0]["rerank_score"] == pytest.approx(0.9)
