"""
Unit tests for IR metrics (Recall@K, nDCG@K, Precision@K, MRR).
Copied from the original test_ir_metrics.py.
"""

from src.cubo.evaluation.metrics import IRMetricsEvaluator


class TestIRMetrics:
    """Test Information Retrieval metrics computation."""

    def test_recall_at_k_perfect(self):
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial(self):
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5"]
        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=4)
        assert abs(recall - 0.6667) < 0.01

    def test_recall_at_k_zero(self):
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc4", "doc5", "doc6"]
        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=3)
        assert recall == 0.0

    def test_precision_at_k_perfect(self):
        relevant = ["doc1", "doc2"]
        retrieved = ["doc1", "doc2"]
        precision = IRMetricsEvaluator.compute_precision_at_k(relevant, retrieved, k=2)
        assert precision == 1.0

    def test_ndcg_at_k_perfect(self):
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        ndcg = IRMetricsEvaluator.compute_ndcg_at_k(relevant, retrieved, k=3)
        assert ndcg == 1.0

    def test_mrr_first_position(self):
        relevant = ["doc2"]
        retrieved = ["doc2", "doc1", "doc3"]
        mrr = IRMetricsEvaluator.compute_mrr(relevant, retrieved)
        assert mrr == 1.0


# Moved from tests/performance/test_ir_metrics.py
