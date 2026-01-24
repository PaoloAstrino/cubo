"""
Quantization-aware routing: Dynamic alpha adaptation based on quantization degradation.

This module provides the mechanism to compute adaptive fusion weights (alpha) that
compensate for quantization error in FAISS IVFPQ indices.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QuantizationRouter:
    """
    Manages adaptive alpha computation based on quantization impact.
    
    Attributes:
        calibration_curve: Dict mapping corpus_id -> calibration params
        alpha_base: Default static alpha value (0.0-1.0)
        use_adaptive: Enable/disable adaptive routing
    """

    def __init__(
        self,
        alpha_base: float = 0.5,
        use_adaptive: bool = True,
        calibration_file: Optional[str] = None,
    ):
        """
        Initialize the quantization router.
        
        Args:
            alpha_base: Base alpha (dense weight) when no quantization detected
            use_adaptive: Whether to enable adaptive alpha computation
            calibration_file: Path to calibration curve JSON file
        """
        self.alpha_base = alpha_base
        self.use_adaptive = use_adaptive
        self.calibration_curve: Dict[str, Dict[str, Any]] = {}

        if calibration_file:
            self.load_calibration(calibration_file)
        else:
            # Try to load from default location
            default_calib_path = Path(__file__).parent.parent / "configs" / "calibration_curves.json"
            if default_calib_path.exists():
                self.load_calibration(str(default_calib_path))

    def load_calibration(self, filepath: str) -> None:
        """
        Load calibration curves from JSON file.
        
        Expected format:
        {
            "corpus_id": {
                "dense_drop_mean": 0.035,
                "beta": 1.75,
                ...
            }
        }
        
        Args:
            filepath: Path to calibration JSON file
        """
        try:
            with open(filepath, "r") as f:
                self.calibration_curve = json.load(f)
            logger.info(f"Loaded calibration curves for {len(self.calibration_curve)} corpora")
        except Exception as e:
            logger.warning(f"Failed to load calibration file {filepath}: {e}")
            self.calibration_curve = {}

    def compute_adaptive_alpha(self, index_metadata: Optional[Dict[str, Any]]) -> float:
        """
        Compute adaptive alpha based on quantization metadata.
        
        Algorithm:
        1. Check if index uses IVFPQ 8-bit quantization
        2. If not quantized, return static alpha
        3. Load corpus-specific calibration curve
        4. Compute: alpha' = alpha_base - (beta * dense_drop)
        5. Clamp to [0, 1]
        
        Args:
            index_metadata: Dict with keys:
                - quantization_type: 'IVFPQ_8bit' or None
                - corpus_id: corpus identifier for lookup
                
        Returns:
            float: Adapted alpha value in [0, 1]
        """
        # Early exit: adaptive disabled or no metadata
        if not self.use_adaptive or index_metadata is None:
            return self.alpha_base

        # Check quantization type
        q_type = index_metadata.get("quantization_type")
        if q_type != "IVFPQ_8bit":
            return self.alpha_base

        # Load corpus-specific calibration
        corpus_id = index_metadata.get("corpus_id")
        if not corpus_id or corpus_id not in self.calibration_curve:
            # Fallback: conservative reduction for unknown corpus
            logger.debug(
                f"Corpus '{corpus_id}' not in calibration curves; using fallback reduction"
            )
            return max(0.0, self.alpha_base - 0.15)

        calib = self.calibration_curve[corpus_id]

        # Extract calibration parameters
        dense_drop = calib.get("dense_drop_mean", 0.0)
        beta = calib.get("beta", 1.75)

        # Compute alpha reduction
        alpha_reduction = beta * dense_drop
        alpha_adapted = max(0.0, min(1.0, self.alpha_base - alpha_reduction))

        logger.debug(
            f"Computed adaptive alpha for '{corpus_id}': "
            f"dense_drop={dense_drop:.4f}, beta={beta:.2f}, "
            f"alpha_base={self.alpha_base:.3f} -> alpha'={alpha_adapted:.3f}"
        )

        return alpha_adapted

    def compute_weights(
        self, index_metadata: Optional[Dict[str, Any]]
    ) -> tuple[float, float]:
        """
        Compute (sparse_weight, dense_weight) tuple from adaptive alpha.
        
        Returns:
            (sparse_weight, dense_weight) where sparse_weight + dense_weight = 1.0
        """
        alpha_adapted = self.compute_adaptive_alpha(index_metadata)
        dense_weight = alpha_adapted
        sparse_weight = 1.0 - alpha_adapted
        return (sparse_weight, dense_weight)

    def enable_adaptive(self, enabled: bool = True) -> None:
        """Enable or disable adaptive routing."""
        self.use_adaptive = enabled
        logger.info(f"Adaptive routing {'enabled' if enabled else 'disabled'}")


# Global instance for convenience
_default_router: Optional[QuantizationRouter] = None


def get_quant_router() -> QuantizationRouter:
    """Get or create the default quantization router."""
    global _default_router
    if _default_router is None:
        _default_router = QuantizationRouter()
    return _default_router
