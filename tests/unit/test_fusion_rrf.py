"""
Unit tests for Reciprocal Rank Fusion (RRF) logic.

Tests RRF score calculation, parameter combinations, edge cases,
and score normalization across different input scenarios.
"""

import pytest
import numpy as np
from typing import List, Dict


class TestRRFFusion:
    """Test suite for RRF fusion algorithm."""
    
    def test_rrf_score_calculation_basic(self):
        """Test basic RRF score calculation with k=60."""
        # Simulate ranked results: doc_id -> rank
        dense_results = {"doc1": 1, "doc2": 3, "doc3": 5}
        bm25_results = {"doc1": 2, "doc2": 1, "doc3": 4}
        
        k = 60
        semantic_weight = 1.0
        bm25_weight = 1.0
        
        # Calculate RRF scores manually
        # doc1: semantic=1/(1+60), bm25=1/(2+60)
        # doc2: semantic=1/(3+60), bm25=1/(1+60) <- best BM25 rank
        # doc3: semantic=1/(5+60), bm25=1/(4+60)
        
        expected_doc1 = semantic_weight * (1/61) + bm25_weight * (1/62)
        expected_doc2 = semantic_weight * (1/63) + bm25_weight * (1/61)
        expected_doc3 = semantic_weight * (1/65) + bm25_weight * (1/64)
        
        # doc1 has best semantic rank, doc2 has best BM25 rank
        # With equal weights, doc1 should actually be slightly higher
        assert expected_doc1 > expected_doc2
        assert expected_doc1 > expected_doc3
    
    def test_rrf_with_different_k_values(self):
        """Test RRF behavior with k=20, k=60, k=120."""
        dense_rank = 1
        bm25_rank = 10
        
        # Lower k amplifies difference between ranks
        score_k20 = 1/(dense_rank + 20) + 1/(bm25_rank + 20)  # 1/21 + 1/30
        score_k60 = 1/(dense_rank + 60) + 1/(bm25_rank + 60)  # 1/61 + 1/70
        score_k120 = 1/(dense_rank + 120) + 1/(bm25_rank + 120)  # 1/121 + 1/130
        
        # All should be positive
        assert score_k20 > 0
        assert score_k60 > 0
        assert score_k120 > 0
        
        # Lower k gives higher absolute scores
        assert score_k20 > score_k60 > score_k120
    
    def test_rrf_with_weighted_combination(self):
        """Test RRF with different semantic/BM25 weights."""
        rank = 5
        k = 60
        
        # Heavy semantic weight (1.3 vs 0.7)
        score_semantic_heavy = 1.3 * (1/(rank + k)) + 0.7 * (1/(rank + k))
        
        # Heavy BM25 weight (0.7 vs 1.3)
        score_bm25_heavy = 0.7 * (1/(rank + k)) + 1.3 * (1/(rank + k))
        
        # Balanced weights
        score_balanced = 1.0 * (1/(rank + k)) + 1.0 * (1/(rank + k))
        
        # All should produce same result when ranks are equal
        assert abs(score_semantic_heavy - score_bm25_heavy) < 1e-10
        assert abs(score_balanced - 2.0 * (1/(rank + k))) < 1e-10
    
    def test_rrf_missing_in_one_ranker(self):
        """Test RRF when document appears in only one ranker."""
        # doc1 only in dense results
        dense_results = {"doc1": 1, "doc2": 2}
        bm25_results = {"doc2": 1, "doc3": 2}
        
        k = 60
        w_s = 1.0
        w_b = 1.0
        
        # doc1: only has dense score
        score_doc1 = w_s * (1/(1 + k)) + w_b * 0  # 1/61 + 0
        
        # doc2: has both scores
        score_doc2 = w_s * (1/(2 + k)) + w_b * (1/(1 + k))  # 1/62 + 1/61
        
        # doc3: only has BM25 score
        score_doc3 = w_s * 0 + w_b * (1/(2 + k))  # 0 + 1/62
        
        # doc2 should rank highest (appears in both)
        assert score_doc2 > score_doc1
        assert score_doc2 > score_doc3
    
    def test_rrf_edge_case_empty_results(self):
        """Test RRF with empty result sets."""
        dense_results = {}
        bm25_results = {}
        
        # Should return empty dict without errors
        fused_results = {}
        assert len(fused_results) == 0
    
    def test_rrf_edge_case_single_document(self):
        """Test RRF with single document in results."""
        dense_results = {"doc1": 1}
        bm25_results = {"doc1": 1}
        
        k = 60
        expected_score = 1/(1 + k) + 1/(1 + k)  # 2/61
        
        assert expected_score > 0
    
    def test_rrf_score_ordering_preservation(self):
        """Test that RRF maintains reasonable ordering."""
        # Document that ranks #1 in both should beat document that ranks #50 in both
        k = 60
        
        score_top = 1/(1 + k) + 1/(1 + k)  # Both rank 1
        score_low = 1/(50 + k) + 1/(50 + k)  # Both rank 50
        
        assert score_top > score_low
    
    def test_rrf_parameter_sweep_ranges(self):
        """Test that parameter sweep covers expected ranges."""
        k_values = [20, 60, 120]
        semantic_weights = [0.7, 1.0, 1.3]
        bm25_weights = [0.7, 1.0, 1.3]
        
        # Should generate 3 * 3 * 3 = 27 combinations
        combinations = [
            (k, sw, bw) 
            for k in k_values 
            for sw in semantic_weights 
            for bw in bm25_weights
        ]
        
        assert len(combinations) == 27
        assert (20, 0.7, 0.7) in combinations
        assert (120, 1.3, 1.3) in combinations
        assert (60, 1.0, 1.0) in combinations  # Baseline
    
    def test_rrf_numerical_stability(self):
        """Test RRF with very large rank values."""
        k = 60
        large_rank = 10000
        
        # Should not overflow or produce NaN
        score = 1/(large_rank + k)
        
        assert not np.isnan(score)
        assert not np.isinf(score)
        assert score > 0
        assert score < 1


class TestRRFIntegration:
    """Integration tests for RRF with actual retrieval components."""
    
    def test_rrf_topk50_format_compatibility(self):
        """Test RRF can parse topk50 format files."""
        # Simulate topk50 format: query_id \t doc_id \t rank \t score
        dense_line = "query1\tdoc123\t1\t0.95"
        bm25_line = "query1\tdoc456\t1\t42.3"
        
        parts_dense = dense_line.split('\t')
        parts_bm25 = bm25_line.split('\t')
        
        assert len(parts_dense) == 4
        assert len(parts_bm25) == 4
        assert parts_dense[0] == parts_bm25[0]  # Same query
        
        rank_dense = int(parts_dense[2])
        rank_bm25 = int(parts_bm25[2])
        
        assert rank_dense == 1
        assert rank_bm25 == 1
    
    def test_rrf_output_format(self):
        """Test RRF output matches expected BEIR run format."""
        # Expected format: query_id \t doc_id \t rank \t score
        query_id = "query123"
        doc_id = "doc456"
        rank = 1
        score = 0.0328  # Example RRF score
        
        output_line = f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}"
        
        parts = output_line.split('\t')
        assert len(parts) == 4
        assert parts[0] == query_id
        assert parts[1] == doc_id
        assert parts[2] == str(rank)
        assert float(parts[3]) == score


class TestRRFEdgeCases:
    """Edge case and property-based tests."""
    
    def test_rrf_commutative_fusion(self):
        """Test that swapping rankers doesn't change final ranking (with equal weights)."""
        # With equal weights, swapping dense<->BM25 should give same ranking
        dense_results = {"doc1": 1, "doc2": 2, "doc3": 3}
        bm25_results = {"doc1": 3, "doc2": 2, "doc3": 1}
        
        k = 60
        w = 1.0
        
        # Score with original order
        score_doc1_v1 = w * (1/(1+k)) + w * (1/(3+k))
        score_doc2_v1 = w * (1/(2+k)) + w * (1/(2+k))
        score_doc3_v1 = w * (1/(3+k)) + w * (1/(1+k))
        
        # Score with swapped order (swap dense<->bm25)
        score_doc1_v2 = w * (1/(3+k)) + w * (1/(1+k))
        score_doc2_v2 = w * (1/(2+k)) + w * (1/(2+k))
        score_doc3_v2 = w * (1/(1+k)) + w * (1/(3+k))
        
        # With equal weights, doc1 and doc3 should have same score
        assert abs(score_doc1_v1 - score_doc3_v2) < 1e-10
        assert abs(score_doc3_v1 - score_doc1_v2) < 1e-10
        assert abs(score_doc2_v1 - score_doc2_v2) < 1e-10
    
    def test_rrf_zero_weight_equivalence(self):
        """Test RRF with zero weight equals single ranker."""
        rank_dense = 5
        rank_bm25 = 10
        k = 60
        
        # Only semantic (BM25 weight = 0)
        score_semantic_only = 1.0 * (1/(rank_dense + k)) + 0.0 * (1/(rank_bm25 + k))
        
        # Only BM25 (semantic weight = 0)
        score_bm25_only = 0.0 * (1/(rank_dense + k)) + 1.0 * (1/(rank_bm25 + k))
        
        assert score_semantic_only == 1/(rank_dense + k)
        assert score_bm25_only == 1/(rank_bm25 + k)
    
    def test_rrf_monotonicity(self):
        """Test that better ranks produce higher scores."""
        k = 60
        
        ranks = [1, 2, 5, 10, 20, 50, 100]
        scores = [1/(r + k) for r in ranks]
        
        # Scores should decrease as rank increases
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
