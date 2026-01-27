"""
Unit tests for quantization-aware routing.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cubo.retrieval.quant_router import QuantizationRouter


@pytest.fixture
def temp_calibration_file():
    """Create a temporary calibration file for testing."""
    calibration = {
        "scifact": {
            "corpus_id": "scifact",
            "dense_drop_mean": 0.035,
            "dense_drop_std": 0.025,
            "beta": 1.75,
            "nlist": 256,
            "nbits": 8,
            "num_queries": 200,
        },
        "fiqa": {
            "corpus_id": "fiqa",
            "dense_drop_mean": 0.045,
            "dense_drop_std": 0.032,
            "beta": 1.75,
            "nlist": 256,
            "nbits": 8,
            "num_queries": 250,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(calibration, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestQuantizationRouter:
    """Test suite for QuantizationRouter."""

    def test_init_default(self):
        """Test default initialization."""
        router = QuantizationRouter()
        assert router.alpha_base == 0.5
        assert router.use_adaptive is True

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        router = QuantizationRouter(alpha_base=0.6, use_adaptive=False)
        assert router.alpha_base == 0.6
        assert router.use_adaptive is False

    def test_load_calibration(self, temp_calibration_file):
        """Test loading calibration from file."""
        router = QuantizationRouter(calibration_file=temp_calibration_file)
        assert "scifact" in router.calibration_curve
        assert "fiqa" in router.calibration_curve
        assert router.calibration_curve["scifact"]["dense_drop_mean"] == 0.035

    def test_adaptive_alpha_with_quantization(self, temp_calibration_file):
        """Verify adaptive α differs from static α when quantization detected."""
        router = QuantizationRouter(
            alpha_base=0.5, use_adaptive=True, calibration_file=temp_calibration_file
        )

        metadata = {
            "quantization_type": "IVFPQ_8bit",
            "corpus_id": "scifact",
            "nlist": 256,
            "nbits": 8,
        }

        alpha_adaptive = router.compute_adaptive_alpha(metadata)

        # Expected: 0.5 - (1.75 * 0.035) = 0.5 - 0.06125 = 0.43875
        assert alpha_adaptive < 0.5  # Should be less than base
        assert abs(alpha_adaptive - 0.439) < 0.01

    def test_static_alpha_without_quantization(self, temp_calibration_file):
        """Verify static α used when no quantization detected."""
        router = QuantizationRouter(
            alpha_base=0.5, use_adaptive=True, calibration_file=temp_calibration_file
        )

        metadata = {"quantization_type": None, "corpus_id": "scifact"}

        alpha = router.compute_adaptive_alpha(metadata)
        assert alpha == 0.5  # Should return static alpha

    def test_static_alpha_when_adaptive_disabled(self, temp_calibration_file):
        """Verify static α used when adaptive disabled."""
        router = QuantizationRouter(
            alpha_base=0.5, use_adaptive=False, calibration_file=temp_calibration_file
        )

        metadata = {
            "quantization_type": "IVFPQ_8bit",
            "corpus_id": "scifact",
        }

        alpha = router.compute_adaptive_alpha(metadata)
        assert alpha == 0.5  # Adaptive disabled, should return static

    def test_fallback_alpha_unknown_corpus(self, temp_calibration_file):
        """Verify fallback reduction for unknown corpus."""
        router = QuantizationRouter(
            alpha_base=0.5, use_adaptive=True, calibration_file=temp_calibration_file
        )

        metadata = {
            "quantization_type": "IVFPQ_8bit",
            "corpus_id": "unknown_corpus",
        }

        alpha = router.compute_adaptive_alpha(metadata)
        # Fallback: max(0.0, 0.5 - 0.15) = 0.35
        assert alpha == 0.35

    def test_alpha_clamping(self, temp_calibration_file):
        """Verify alpha is clamped to [0, 1]."""
        # Create router with very high alpha_base
        router = QuantizationRouter(
            alpha_base=0.95, use_adaptive=True, calibration_file=temp_calibration_file
        )

        metadata = {
            "quantization_type": "IVFPQ_8bit",
            "corpus_id": "scifact",
        }

        alpha = router.compute_adaptive_alpha(metadata)
        assert 0.0 <= alpha <= 1.0

    def test_compute_weights(self, temp_calibration_file):
        """Test conversion of alpha to (sparse_weight, dense_weight)."""
        router = QuantizationRouter(
            alpha_base=0.5, use_adaptive=True, calibration_file=temp_calibration_file
        )

        metadata = {
            "quantization_type": "IVFPQ_8bit",
            "corpus_id": "scifact",
        }

        sparse_weight, dense_weight = router.compute_weights(metadata)

        # Weights should sum to 1.0
        assert abs(sparse_weight + dense_weight - 1.0) < 1e-6
        # Dense should be smaller (quantization degradation)
        assert dense_weight < 0.5

    def test_enable_disable_adaptive(self):
        """Test enabling/disabling adaptive routing."""
        router = QuantizationRouter(use_adaptive=True)
        assert router.use_adaptive is True

        router.enable_adaptive(False)
        assert router.use_adaptive is False

        router.enable_adaptive(True)
        assert router.use_adaptive is True

    def test_none_metadata(self):
        """Verify None metadata returns static alpha."""
        router = QuantizationRouter(alpha_base=0.5, use_adaptive=True)
        alpha = router.compute_adaptive_alpha(None)
        assert alpha == 0.5

    def test_different_alpha_bases(self, temp_calibration_file):
        """Test behavior with different alpha_base values."""
        for alpha_base in [0.3, 0.5, 0.7]:
            router = QuantizationRouter(
                alpha_base=alpha_base, use_adaptive=True, calibration_file=temp_calibration_file
            )

            metadata = {
                "quantization_type": "IVFPQ_8bit",
                "corpus_id": "scifact",
            }

            alpha_adapted = router.compute_adaptive_alpha(metadata)
            # Adapted should be less than base (quantization reduces alpha)
            assert alpha_adapted < alpha_base
            # But should still be valid
            assert 0.0 <= alpha_adapted <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    def test_adaptive_alpha_with_degradation(self):
        """When quant-aware is enabled, α should be adjusted based on degradation."""
        router = QueryRouter(
            presets={
                "factual": {
                    "bm25_weight": 0.5,
                    "dense_weight": 0.5,
                    "use_reranker": False,
                    "k_candidates": 100,
                }
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

        monkeypatch.setitem(
            config._overrides,
            "query_router",
            {"quant_sensitivity": 2.0},  # More aggressive adjustment
        )

        router = QueryRouter(
            presets={
                "factual": {
                    "bm25_weight": 0.5,
                    "dense_weight": 0.5,
                    "use_reranker": False,
                    "k_candidates": 100,
                }
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
                "factual": {
                    "bm25_weight": 0.6,
                    "dense_weight": 0.4,
                    "use_reranker": False,
                    "k_candidates": 100,
                }
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
                "conceptual": {
                    "bm25_weight": 0.2,
                    "dense_weight": 0.8,
                    "use_reranker": True,
                    "k_candidates": 100,
                }
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
                "factual": {
                    "bm25_weight": 0.6,
                    "dense_weight": 0.4,
                    "use_reranker": False,
                    "k_candidates": 100,
                },
                "conceptual": {
                    "bm25_weight": 0.3,
                    "dense_weight": 0.7,
                    "use_reranker": True,
                    "k_candidates": 100,
                },
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
