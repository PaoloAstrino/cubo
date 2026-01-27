import pytest

from evaluation import ragas_evaluator as rev


def test_run_ragas_evaluation_serial_aggregation(monkeypatch):
    # Prepare tiny synthetic dataset
    questions = ["Q1", "Q2"]
    contexts = [["ctx1a", "ctx1b"], ["ctx2a"]]
    ground_truths = ["GT1", "GT2"]
    answers = ["A1", "A2"]

    # Dummy evaluate implementation that returns different scores per-call
    call_counter = {"count": 0}

    def fake_evaluate(dataset, metrics, llm, embeddings):
        # Each call returns a dict with two numeric metrics
        call_counter["count"] += 1
        return {"faithfulness": 0.5 + 0.1 * call_counter["count"], "context_precision": 0.8}

    monkeypatch.setattr(rev, "evaluate", fake_evaluate)

    # Run in serial mode to force per-sample invocation
    out = rev.run_ragas_evaluation(
        questions, contexts, ground_truths, answers, llm=None, max_workers=1
    )

    # Expect averaged faithfulness: (0.6 + 0.7)/2 = 0.65
    assert pytest.approx(out.get("faithfulness", 0.0), rel=1e-6) == 0.65
    assert out.get("context_precision") == pytest.approx(0.8)
    # Ensure evaluate was called exactly twice
    assert call_counter["count"] == 2
