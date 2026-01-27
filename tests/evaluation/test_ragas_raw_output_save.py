"""Test per-sample raw output saving in RAGAS evaluation."""

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.ragas_evaluator import run_ragas_evaluation


def test_per_sample_raw_output_saved(monkeypatch, tmp_path):
    """Test that per-sample raw outputs are written to JSONL when enabled."""
    # Prepare synthetic inputs
    questions = ["Q1", "Q2"]
    contexts = [["ctx1a", "ctx1b"], ["ctx2a"]]
    ground_truths = ["GT1", "GT2"]
    answers = ["A1", "A2"]
    retrieval_times = [0.1, 0.2]
    generation_times = [0.5, 0.6]

    # Mock the evaluate function to return a simple dict result
    from evaluation import ragas_evaluator as rev

    def fake_evaluate(dataset, metrics, llm, embeddings):
        # Return a mock result with aggregate scores
        class FakeResult:
            def to_dict(self):
                return {"faithfulness": 0.8, "context_precision": 0.9}

        return FakeResult()

    monkeypatch.setattr(rev, "evaluate", fake_evaluate)

    # Define output path
    output_path = tmp_path / "raw_output.jsonl"

    # Run evaluation with per-sample saving enabled
    scores = run_ragas_evaluation(
        questions=questions,
        contexts=contexts,
        ground_truths=ground_truths,
        answers=answers,
        llm=None,  # Will use default local LLM
        save_per_sample_path=str(output_path),
        retrieval_times=retrieval_times,
        generation_times=generation_times,
    )

    # Verify aggregate scores returned
    assert "faithfulness" in scores
    assert scores["faithfulness"] == 0.8

    # Verify JSONL file created and contains expected fields
    assert output_path.exists()

    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2  # Two samples

        for i, line in enumerate(lines):
            sample = json.loads(line)
            assert sample["sample_id"] == i
            assert sample["question"] == questions[i]
            assert sample["contexts"] == contexts[i]
            assert sample["ground_truth"] == ground_truths[i]
            assert sample["answer"] == answers[i]
            assert sample["retrieval_time"] == retrieval_times[i]
            assert sample["generation_time"] == generation_times[i]


def test_per_sample_raw_output_optional(monkeypatch, tmp_path):
    """Test that per-sample raw output saving is truly optional (default off)."""
    questions = ["Q1"]
    contexts = [["ctx1"]]
    ground_truths = ["GT1"]
    answers = ["A1"]

    from evaluation import ragas_evaluator as rev

    def fake_evaluate(dataset, metrics, llm, embeddings):
        class FakeResult:
            def to_dict(self):
                return {"faithfulness": 0.7}

        return FakeResult()

    monkeypatch.setattr(rev, "evaluate", fake_evaluate)

    # Run without specifying save_per_sample_path
    scores = run_ragas_evaluation(
        questions=questions,
        contexts=contexts,
        ground_truths=ground_truths,
        answers=answers,
        llm=None,
    )

    # Should still return aggregate scores
    assert "faithfulness" in scores

    # No JSONL file should be created
    assert not any(tmp_path.glob("*.jsonl"))
