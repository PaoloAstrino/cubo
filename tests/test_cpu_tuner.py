import os
from unittest.mock import MagicMock, patch
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
        allocator="libc"
    )

def test_auto_tune_cpu_dry_run(mock_profile):
    """Test that dry run returns changes but doesn't modify env."""
    with patch.dict(os.environ, {}, clear=True):
        changes = cpu_tuner.auto_tune_cpu(mock_profile, dry_run=True)
        
        # Should recommend setting threads to 7 (8 physical - 1 reserved)
        assert changes["OMP_NUM_THREADS"] == "7"
        assert "OMP_NUM_THREADS" not in os.environ

def test_auto_tune_cpu_apply(mock_profile):
    """Test that apply actually modifies env."""
    with patch.dict(os.environ, {}, clear=True):
        changes = cpu_tuner.auto_tune_cpu(mock_profile, dry_run=False)
        
        assert os.environ["OMP_NUM_THREADS"] == "7"
        assert changes["OMP_NUM_THREADS"] == "7"

def test_auto_tune_cpu_respects_existing(mock_profile):
    """Test that existing env vars are not overwritten."""
    with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}, clear=True):
        changes = cpu_tuner.auto_tune_cpu(mock_profile, dry_run=False)
        
        assert "OMP_NUM_THREADS" not in changes
        assert os.environ["OMP_NUM_THREADS"] == "1"
        # Other vars should still be set
        assert changes["MKL_NUM_THREADS"] == "7"

def test_auto_tune_low_cores():
    """Test heuristic for low core count machines."""
    profile = HardwareProfile(
        device="cpu", n_gpu_layers=0, vram_gb=0,
        physical_cores=4, logical_cores=8, total_ram_gb=8
    )
    with patch.dict(os.environ, {}, clear=True):
        changes = cpu_tuner.auto_tune_cpu(profile, dry_run=True)
        assert changes["OMP_NUM_THREADS"] == "4" # No reservation for low core count

def test_mkl_runtime_call(mock_profile):
    """Test that mkl.set_num_threads is called if backend is mkl."""
    with patch.dict(os.environ, {}, clear=True):
        
        # Setup mock mkl module
        mock_mkl = MagicMock()
        
        # We need to patch sys.modules so that 'import mkl' returns our mock
        with patch.dict("sys.modules", {"mkl": mock_mkl}):
            cpu_tuner.auto_tune_cpu(mock_profile, dry_run=False)
            mock_mkl.set_num_threads.assert_called_with(7)
