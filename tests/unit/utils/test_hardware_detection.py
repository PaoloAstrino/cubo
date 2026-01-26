import sys
from unittest.mock import MagicMock, patch

from cubo.utils import cpu_features, hardware


def test_get_cpu_flags_with_cpuinfo():
    """Test that cpu flags are retrieved from cpuinfo."""
    mock_cpuinfo = MagicMock()
    mock_cpuinfo.get_cpu_info.return_value = {"flags": ["avx", "avx2", "sse"]}

    with patch.dict(sys.modules, {"cpuinfo": mock_cpuinfo}):
        # Reload module to pick up the mock if it was already imported differently
        # But since we import cpu_features, we can patch the module attribute inside it
        with patch("cubo.utils.cpu_features.cpuinfo", mock_cpuinfo):
            flags = cpu_features.get_cpu_flags()
            assert "avx" in flags
            assert "avx2" in flags
            assert "sse" in flags


def test_get_cpu_flags_missing_cpuinfo():
    """Test graceful fallback when cpuinfo is missing."""
    with patch("cubo.utils.cpu_features.cpuinfo", None):
        flags = cpu_features.get_cpu_flags()
        assert flags == []


def test_get_topology_psutil():
    """Test topology detection via psutil."""
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.side_effect = lambda logical: 8 if logical else 4

    with patch("cubo.utils.cpu_features.psutil", mock_psutil):
        topo = cpu_features.get_topology()
        assert topo["physical_cores"] == 4
        assert topo["logical_cores"] == 8


def test_detect_blas_backend_numpy():
    """Test BLAS detection via numpy config."""
    mock_np = MagicMock()
    # Configure the mock to have __config__ attribute
    # We need to attach it explicitly because MagicMock might treat it specially
    config_mock = MagicMock()
    config_mock.get_info.side_effect = lambda x: (
        {"libraries": ["mkl_rt"]} if x == "mkl_info" else {}
    )

    # We can't assign to __config__ directly on a MagicMock sometimes if it's treated as magic
    # But we can pass it in constructor or configure_mock
    mock_np.configure_mock(**{"__config__": config_mock})

    with patch("cubo.utils.cpu_features.np", mock_np):
        backend, _ = cpu_features.detect_blas_backend()
        assert backend == "mkl"


def test_detect_hardware_integration():
    """Test the full detect_hardware function integrates new fields."""
    # Mock torch to avoid GPU checks
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ), patch("cubo.utils.cpu_features.get_cpu_flags", return_value=["avx2"]), patch(
        "cubo.utils.cpu_features.get_topology",
        return_value={"physical_cores": 4, "logical_cores": 8},
    ), patch(
        "cubo.utils.cpu_features.detect_blas_backend", return_value=("openblas", {})
    ), patch(
        "psutil.virtual_memory"
    ) as mock_vm:

        mock_vm.return_value.total = 16 * (1024**3)  # 16 GB

        profile = hardware.detect_hardware()

        assert profile.device == "cpu"
        assert profile.physical_cores == 4
        assert profile.logical_cores == 8
        assert profile.total_ram_gb == 16.0
        assert "avx2" in profile.cpu_flags
        assert profile.blas_backend == "openblas"
