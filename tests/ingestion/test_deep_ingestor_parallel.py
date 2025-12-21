import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.utils.hardware import HardwareProfile

@pytest.fixture
def mock_hardware_8_cores():
    return HardwareProfile(
        device="cpu", n_gpu_layers=0, vram_gb=0,
        physical_cores=8, logical_cores=16, total_ram_gb=16,
        cpu_flags=[], blas_backend="mkl", allocator="libc"
    )

def test_apply_laptop_mode_dynamic_workers(mock_hardware_8_cores):
    """Test that apply_laptop_mode sets n_workers based on physical cores."""
    from cubo.config import config
    
    with patch("cubo.utils.hardware.detect_hardware", return_value=mock_hardware_8_cores):
        # Force enable laptop mode
        with patch.object(config, "is_laptop_mode", return_value=True):
            config.apply_laptop_mode(force=True)
            
            # Should be 8 - 1 = 7
            assert config.get("ingestion.deep.n_workers") == 7

def test_deep_ingestor_uses_parallel_execution():
    """Test that DeepIngestor uses ProcessPoolExecutor when n_workers > 1."""
    with patch("cubo.ingestion.deep_ingestor.ProcessPoolExecutor") as mock_executor_cls, \
         patch("cubo.ingestion.deep_ingestor.as_completed") as mock_as_completed, \
         patch("cubo.ingestion.deep_ingestor.get_metadata_manager"), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("os.walk", return_value=[("/tmp", [], ["file1.txt", "file2.txt"])]), \
         patch("cubo.ingestion.deep_ingestor.DeepIngestor._process_file", return_value=[{"text": "foo"}]):
        
        # Setup mocks
        mock_executor = mock_executor_cls.return_value
        mock_executor.__enter__.return_value = mock_executor
        
        # Mock futures
        f1, f2 = MagicMock(), MagicMock()
        f1.result.return_value = [{"text": "chunk1"}]
        f2.result.return_value = [{"text": "chunk2"}]
        
        mock_executor.submit.side_effect = [f1, f2]
        mock_as_completed.return_value = [f1, f2]
        
        # Init ingestor with n_workers=2
        ingestor = DeepIngestor(input_folder="/tmp", output_dir="/tmp", n_workers=2)
        
        # Run ingest
        ingestor.ingest()
        
        # Verify executor usage
        mock_executor_cls.assert_called_with(max_workers=2)
        assert mock_executor.submit.call_count == 2

def test_deep_ingestor_serial_fallback():
    """Test that DeepIngestor uses serial loop when n_workers=1."""
    with patch("cubo.ingestion.deep_ingestor.ProcessPoolExecutor") as mock_executor_cls, \
         patch("cubo.ingestion.deep_ingestor.get_metadata_manager"), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("os.walk", return_value=[("/tmp", [], ["file1.txt"])]), \
         patch("cubo.ingestion.deep_ingestor.DeepIngestor._process_file", return_value=[{"text": "foo"}]):
        
        ingestor = DeepIngestor(input_folder="/tmp", output_dir="/tmp", n_workers=1)
        ingestor.ingest()
        
        # Should NOT use executor
        mock_executor_cls.assert_not_called()
