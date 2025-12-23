from unittest.mock import patch

import pytest

from cubo.embeddings.model_loader import ModelManager
from cubo.utils.hardware import HardwareProfile


@pytest.fixture
def mock_hardware_avx2():
    return HardwareProfile(
        device="cpu",
        n_gpu_layers=0,
        vram_gb=0,
        physical_cores=4,
        logical_cores=8,
        total_ram_gb=16,
        cpu_flags=["avx2"],
        blas_backend="mkl",
        allocator="libc",
    )


@pytest.fixture
def mock_hardware_no_avx():
    return HardwareProfile(
        device="cpu",
        n_gpu_layers=0,
        vram_gb=0,
        physical_cores=4,
        logical_cores=8,
        total_ram_gb=16,
        cpu_flags=[],
        blas_backend="openblas",
        allocator="libc",
    )


def test_model_loader_prefers_quantized_auto(mock_hardware_avx2):
    """Test that loader attempts optimization when AVX2 is present and config is auto."""
    # Patch where it is defined, not where it is imported locally
    with patch("cubo.utils.hardware.detect_hardware", return_value=mock_hardware_avx2), patch(
        "cubo.embeddings.model_loader.config.get"
    ) as mock_config, patch("cubo.monitoring.metrics.record") as mock_record, patch(
        "sentence_transformers.SentenceTransformer"
    ):

        # Setup config
        def config_side_effect(key, default=None):
            if key == "embeddings.prefer_quantized_cpu":
                return "auto"
            if key == "model_path":
                return "test-model"
            if key == "embeddings.device":
                return "cpu"
            return default

        mock_config.side_effect = config_side_effect

        manager = ModelManager(lazy=False)
        manager.load_model()

        # Should have recorded usage of quantized model
        mock_record.assert_called_with("model_quantized_used", 1)


def test_model_loader_skips_quantized_no_avx(mock_hardware_no_avx):
    """Test that loader skips optimization when AVX is missing."""
    with patch("cubo.utils.hardware.detect_hardware", return_value=mock_hardware_no_avx), patch(
        "cubo.embeddings.model_loader.config.get"
    ) as mock_config, patch("cubo.monitoring.metrics.record") as mock_record, patch(
        "sentence_transformers.SentenceTransformer"
    ):

        def config_side_effect(key, default=None):
            if key == "embeddings.prefer_quantized_cpu":
                return "auto"
            if key == "model_path":
                return "test-model"
            if key == "embeddings.device":
                return "cpu"
            return default

        mock_config.side_effect = config_side_effect

        manager = ModelManager(lazy=False)
        manager.load_model()

        # Should NOT have recorded usage
        mock_record.assert_not_called()


def test_model_loader_force_quantized(mock_hardware_no_avx):
    """Test that loader attempts optimization if forced, even without AVX."""
    with patch("cubo.utils.hardware.detect_hardware", return_value=mock_hardware_no_avx), patch(
        "cubo.embeddings.model_loader.config.get"
    ) as mock_config, patch("cubo.monitoring.metrics.record") as mock_record, patch(
        "sentence_transformers.SentenceTransformer"
    ):

        def config_side_effect(key, default=None):
            if key == "embeddings.prefer_quantized_cpu":
                return "always"
            if key == "model_path":
                return "test-model"
            if key == "embeddings.device":
                return "cpu"
            return default

        mock_config.side_effect = config_side_effect

        manager = ModelManager(lazy=False)
        manager.load_model()

        mock_record.assert_called_with("model_quantized_used", 1)
