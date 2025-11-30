"""
Unit tests for IR metrics (Recall@K, nDCG@K, Precision@K, MRR).
"""

import pytest

from cubo.evaluation.metrics import IRMetricsEvaluator


class TestIRMetrics:
    """Test Information Retrieval metrics computation."""

    def test_recall_at_k_perfect(self):
        """Test Recall@K with perfect retrieval."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=5)
        assert recall == 1.0, "Perfect recall should be 1.0"

    def test_recall_at_k_partial(self):
        """Test Recall@K with partial retrieval."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5"]

        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=4)
        assert abs(recall - 0.6667) < 0.01, "Recall should be ~0.67 (2 out of 3)"

    def test_recall_at_k_zero(self):
        """Test Recall@K with no relevant docs retrieved."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc4", "doc5", "doc6"]

        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=3)
        assert recall == 0.0, "Recall should be 0 when no relevant docs retrieved"

    def test_recall_at_k_empty_relevant(self):
        """Test Recall@K with empty relevant list."""
        relevant = []
        retrieved = ["doc1", "doc2"]

        recall = IRMetricsEvaluator.compute_recall_at_k(relevant, retrieved, k=2)
        assert recall == 0.0, "Recall should be 0 for empty relevant list"

    def test_precision_at_k_perfect(self):
        """Test Precision@K with all retrieved docs relevant."""
        relevant = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        retrieved = ["doc1", "doc2", "doc3"]

        precision = IRMetricsEvaluator.compute_precision_at_k(relevant, retrieved, k=3)
        assert precision == 1.0, "Precision should be 1.0 when all retrieved are relevant"

    def test_precision_at_k_partial(self):
        """Test Precision@K with some irrelevant docs."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2", "doc5"]

        precision = IRMetricsEvaluator.compute_precision_at_k(relevant, retrieved, k=4)
        assert precision == 0.5, "Precision should be 0.5 (2 out of 4)"

    def test_precision_at_k_zero(self):
        """Test Precision@K with no relevant docs."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]

        precision = IRMetricsEvaluator.compute_precision_at_k(relevant, retrieved, k=3)
        assert precision == 0.0, "Precision should be 0 when no relevant docs retrieved"

    def test_ndcg_at_k_perfect_ranking(self):
        """Test nDCG@K with perfect ranking."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4"]

        ndcg = IRMetricsEvaluator.compute_ndcg_at_k(relevant, retrieved, k=3)
        assert ndcg == 1.0, "nDCG should be 1.0 for perfect ranking"

    def test_ndcg_at_k_imperfect_ranking(self):
        """Test nDCG@K with imperfect ranking."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc4", "doc1", "doc2", "doc3"]

        ndcg = IRMetricsEvaluator.compute_ndcg_at_k(relevant, retrieved, k=4)
        # nDCG should be less than 1.0 due to imperfect ranking
        assert 0.0 < ndcg < 1.0, "nDCG should be between 0 and 1 for imperfect ranking"

    def test_ndcg_at_k_with_scores(self):
        """Test nDCG@K with custom relevance scores."""
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc4", "doc2"]
        relevance_scores = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}

        ndcg = IRMetricsEvaluator.compute_ndcg_at_k(
            relevant, retrieved, k=3, relevance_scores=relevance_scores
        )
        assert 0.0 <= ndcg <= 1.0, "nDCG should be normalized between 0 and 1"

    def test_mrr_first_position(self):
        """Test MRR when first relevant doc is at position 1."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc1", "doc3", "doc4"]

        mrr = IRMetricsEvaluator.compute_mrr(relevant, retrieved)
        assert mrr == 1.0, "MRR should be 1.0 when first relevant doc is at position 1"

    def test_mrr_second_position(self):
        """Test MRR when first relevant doc is at position 2."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc3", "doc1", "doc4"]

        mrr = IRMetricsEvaluator.compute_mrr(relevant, retrieved)
        assert mrr == 0.5, "MRR should be 0.5 when first relevant doc is at position 2"

    def test_mrr_not_found(self):
        """Test MRR when no relevant docs retrieved."""
        relevant = ["doc1", "doc2"]
        retrieved = ["doc3", "doc4", "doc5"]

        mrr = IRMetricsEvaluator.compute_mrr(relevant, retrieved)
        assert mrr == 0.0, "MRR should be 0 when no relevant docs retrieved"

    def test_evaluate_retrieval_complete(self):
        """Test complete retrieval evaluation."""
        question_id = "q1"
        relevant_ids = ["doc1", "doc2", "doc3"]
        retrieved_ids = ["doc1", "doc4", "doc2", "doc5", "doc3"]
        ground_truth = {"q1": relevant_ids}

        results = IRMetricsEvaluator.evaluate_retrieval(
            question_id, retrieved_ids, ground_truth, k_values=[3, 5]
        )

        assert "recall_at_k" in results
        assert "precision_at_k" in results
        assert "ndcg_at_k" in results
        assert "mrr" in results

        assert 3 in results["recall_at_k"]
        assert 5 in results["recall_at_k"]

        # At k=5, should retrieve all 3 relevant docs
        assert results["recall_at_k"][5] == 1.0

    def test_evaluate_retrieval_no_ground_truth(self):
        """Test retrieval evaluation with missing ground truth."""
        question_id = "q_unknown"
        retrieved_ids = ["doc1", "doc2", "doc3"]
        ground_truth = {"q1": ["doc1", "doc2"]}

        results = IRMetricsEvaluator.evaluate_retrieval(
            question_id, retrieved_ids, ground_truth, k_values=[5]
        )

        assert "error" in results
        assert results["error"] == "no_ground_truth"
        assert results["recall_at_k"][5] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
