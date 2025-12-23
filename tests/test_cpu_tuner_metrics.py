import os
from unittest.mock import patch

import pytest

from cubo.utils import cpu_tuner
from cubo.utils.hardware import HardwareProfile


@pytest.fixture
def mock_profile():
    return HardwareProfile(
        device="cpu",
        n_gpu_layers=0,
        vram_gb=0.0,
        physical_cores=8,
        logical_cores=16,
        total_ram_gb=16.0,
        cpu_flags=["avx2"],
        blas_backend="mkl",
        allocator="libc",
    )


def test_auto_tune_records_metrics(mock_profile):
    """Test that tuning records metrics."""
    with patch.dict(os.environ, {}, clear=True), patch(
        "cubo.utils.cpu_tuner.metrics.record"
    ) as mock_record:

        cpu_tuner.auto_tune_cpu(mock_profile, dry_run=False)

        mock_record.assert_called_with("cpu_tuning_applied", 1)


def test_auto_tune_no_metrics_on_dry_run(mock_profile):
    """Test that dry run does not record metrics."""
    with patch.dict(os.environ, {}, clear=True), patch(
        "cubo.utils.cpu_tuner.metrics.record"
    ) as mock_record:

        cpu_tuner.auto_tune_cpu(mock_profile, dry_run=True)

        mock_record.assert_not_called()
