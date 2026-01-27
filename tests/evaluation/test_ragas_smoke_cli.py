import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import pytest

# Use existing mocks and monkeypatch patterns from test_ragas_integration


def test_ragas_smoke_cli(monkeypatch, tmp_path):
    # Prepare a small synthetic query set and ensure CuboCore returns deterministic outputs
    from tools import run_generation_eval as runmod

    # Monkeypatch load_test_queries to return 2 synthetic queries
    def _load_test_queries(name, n):
        return [
            {
                "question": "What is GDPR Article 5?",
                "ground_truth": "Principles for processing personal data.",
                "query_id": "q1",
            },
            {
                "question": "What is the role of the European Parliament?",
                "ground_truth": "Legislative body of the EU.",
                "query_id": "q2",
            },
        ]

    monkeypatch.setattr(runmod, "load_test_queries", _load_test_queries)

    # Monkeypatch run_rag_pipeline to return simple deterministic contexts/answers with retrieval/generation times
    def _run_rag_pipeline(cubo, question, args):
        return {
            "contexts": ["Span 1 supporting.", "Span 2 supporting."],
            "answer": "Mocked answer.",
            "retrieval_time": 0.1,
            "generation_time": 0.2,
        }

    monkeypatch.setattr(runmod, "run_rag_pipeline", _run_rag_pipeline)

    # Monkeypatch ragas evaluator to return fixed scores to avoid heavy deps
    import evaluation.ragas_evaluator as rev

    def _fake_run_ragas_evaluation(
        questions,
        contexts,
        ground_truths,
        answers,
        llm=None,
        max_workers=None,
        save_per_sample_path=None,
        retrieval_times=None,
        generation_times=None,
    ):
        return {"comprehensiveness": 0.8, "diversity": 0.7, "empowerment": 0.9}

    monkeypatch.setattr(rev, "run_ragas_evaluation", _fake_run_ragas_evaluation)

    # Run CLI module as script, expecting it to write results file
    out_dir = tmp_path / "ragas_out"
    sys_argv = [
        "run_generation_eval.py",
        "--dataset",
        "politics",
        "--num-samples",
        "2",
        "--output",
        str(out_dir),
        "--judge",
        "local",
    ]
    monkeypatch.setattr(sys, "argv", sys_argv)

    # Execute module main
    runpy.run_module("tools.run_generation_eval", run_name="__main__")

    # Check that results file exists and contains ragas_scores
    result_file = out_dir / "politics_ragas_results.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    assert "ragas_scores" in data
    assert data["ragas_scores"]["comprehensiveness"] == 0.8
