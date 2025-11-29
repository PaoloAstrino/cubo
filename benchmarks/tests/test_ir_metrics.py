import os
import sys
import pytest

# Ensure repo root in sys.path so `benchmarks` package resolves during pytest
sys.path.insert(0, os.getcwd())
from benchmarks.retrieval.rag_benchmark import IRMetricsEvaluator


def test_mrr_single_relevant_first():
    evaluator = IRMetricsEvaluator()
    ground_truth = {"q1": ["d3", "d2"]}
    retrieved = ["d3", "d5", "d2"]
    metrics = evaluator.evaluate_retrieval("q1", retrieved, ground_truth, k_values=[5, 10])
    assert "mrr" in metrics
    assert metrics["mrr"] == pytest.approx(1.0)  # first relevant doc at rank 1


def test_mrr_single_relevant_later():
    evaluator = IRMetricsEvaluator()
    ground_truth = {"q1": ["d2"]}
    retrieved = ["d5", "d3", "d2"]
    metrics = evaluator.evaluate_retrieval("q1", retrieved, ground_truth, k_values=[5, 10])
    assert "mrr" in metrics
    assert metrics["mrr"] == pytest.approx(1.0 / 3.0)


def test_mrr_no_relevant():
    evaluator = IRMetricsEvaluator()
    ground_truth = {"q1": ["d10"]}
    retrieved = ["d1", "d2", "d3"]
    metrics = evaluator.evaluate_retrieval("q1", retrieved, ground_truth, k_values=[5, 10])
    assert "mrr" in metrics
    assert metrics["mrr"] == pytest.approx(0.0)
