"""Tests for quantization-aware routing."""

import pytest

from cubo.routing.query_router import QueryRouter, QueryType, RetrievalStrategy


class TestQuantizationAwareRouting:
    """Test adaptive α adjustment based on quantization degradation."""

    def test_quant_aware_disabled_by_default(self):
        """Quant-aware routing should be disabled by default."""
        router = QueryRouter()
        assert router.quant_aware_enabled is False
        assert router.quant_degradation_factor == 0.0

    def test_static_alpha_no_adjustment(self):
        """When quant-aware is disabled, α should not be adjusted."""
        router = QueryRouter(
            presets={
                "factual": {"bm25_weight": 0.6, "dense_weight": 0.4, "use_reranker": False, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = False
        
        strategy = router.compute_strategy("What is the capital of France?")
        
        # Should use static preset values
        assert strategy.bm25_weight == 0.6
        assert strategy.dense_weight == 0.4

    def test_adaptive_alpha_with_degradation(self):
        """When quant-aware is enabled, α should be adjusted based on degradation."""
        router = QueryRouter(
            presets={
                "factual": {"bm25_weight": 0.5, "dense_weight": 0.5, "use_reranker": False, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = True
        router.quant_degradation_factor = 0.10  # 10% recall drop
        
        strategy = router.compute_strategy("What is the capital of France?")
        
        # Dense weight should be reduced: 0.5 * (1 - 1.0 * 0.10) = 0.45
        # After renormalization: dense = 0.45/(0.5+0.45) = 0.474
        assert strategy.dense_weight < 0.5
        assert strategy.bm25_weight > 0.5
        # Weights should sum to 1.0
        assert abs((strategy.bm25_weight + strategy.dense_weight) - 1.0) < 0.001

    def test_sensitivity_parameter(self, monkeypatch):
        """Test different sensitivity (β) values."""
        from cubo.config import config
        
        monkeypatch.setitem(config._data, "query_router", {
            "quant_sensitivity": 2.0  # More aggressive adjustment
        })
        
        router = QueryRouter(
            presets={
                "factual": {"bm25_weight": 0.5, "dense_weight": 0.5, "use_reranker": False, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = True
        router.quant_degradation_factor = 0.10
        
        strategy = router.compute_strategy("What is the capital of France?")
        
        # With β=2.0: 0.5 * (1 - 2.0 * 0.10) = 0.5 * 0.8 = 0.4
        # More aggressive reduction in dense weight
        assert strategy.dense_weight < 0.45

    def test_zero_degradation_no_change(self):
        """With zero degradation, α should not change."""
        router = QueryRouter(
            presets={
                "factual": {"bm25_weight": 0.6, "dense_weight": 0.4, "use_reranker": False, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = True
        router.quant_degradation_factor = 0.0  # No degradation
        
        strategy = router.compute_strategy("What is the capital of France?")
        
        # Should keep original weights
        assert abs(strategy.bm25_weight - 0.6) < 0.01
        assert abs(strategy.dense_weight - 0.4) < 0.01

    def test_high_degradation_caps_at_zero(self):
        """Extreme degradation should not produce negative weights."""
        router = QueryRouter(
            presets={
                "conceptual": {"bm25_weight": 0.2, "dense_weight": 0.8, "use_reranker": True, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = True
        router.quant_degradation_factor = 0.95  # 95% degradation (extreme)
        
        strategy = router.compute_strategy("Why does the sky appear blue?")
        
        # Dense weight should be heavily reduced but not negative
        assert strategy.dense_weight >= 0.0
        assert strategy.bm25_weight >= 0.0
        assert abs((strategy.bm25_weight + strategy.dense_weight) - 1.0) < 0.001

    def test_different_query_types_adjusted(self):
        """All query types should respect quant-aware adjustment."""
        router = QueryRouter(
            presets={
                "factual": {"bm25_weight": 0.6, "dense_weight": 0.4, "use_reranker": False, "k_candidates": 100},
                "conceptual": {"bm25_weight": 0.3, "dense_weight": 0.7, "use_reranker": True, "k_candidates": 100}
            }
        )
        router.quant_aware_enabled = True
        router.quant_degradation_factor = 0.15
        
        factual_strategy = router.compute_strategy("What is GDP?")
        conceptual_strategy = router.compute_strategy("Why does inflation occur?")
        
        # Both should have reduced dense weights
        assert factual_strategy.dense_weight < 0.4
        assert conceptual_strategy.dense_weight < 0.7
        # But conceptual should still favor dense more than factual
        assert conceptual_strategy.dense_weight > factual_strategy.dense_weight


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
