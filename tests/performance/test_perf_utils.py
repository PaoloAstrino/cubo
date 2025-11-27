"""
Unit tests for performance utilities (latency, memory sampling, hardware metadata).
"""

import time

import pytest

from src.cubo.evaluation.perf_utils import (
    format_hardware_summary,
    log_hardware_metadata,
    sample_latency,
    sample_memory,
)


class TestPerfUtils:
    """Test performance measurement utilities."""

    def test_sample_latency_single(self):
        """Test latency sampling with single sample."""

        def slow_function(duration):
            time.sleep(duration)
            return "done"

        metrics = sample_latency(slow_function, 0.01, samples=1)

        assert "mean_ms" in metrics
        assert "p50_ms" in metrics
        assert "p95_ms" in metrics
        assert "p99_ms" in metrics
        assert "samples" in metrics

        assert metrics["samples"] == 1
        assert metrics["mean_ms"] >= 10.0  # At least 10ms for 0.01s sleep

    def test_sample_latency_multiple(self):
        """Test latency sampling with multiple samples."""

        def fast_function():
            return sum(range(1000))

        metrics = sample_latency(fast_function, samples=5)

        assert metrics["samples"] == 5
        assert len(metrics["all_samples_ms"]) == 5
        assert metrics["min_ms"] <= metrics["mean_ms"] <= metrics["max_ms"]
        assert metrics["p50_ms"] >= 0

    def test_sample_latency_with_args(self):
        """Test latency sampling with function arguments."""

        def add_numbers(a, b):
            return a + b

        metrics = sample_latency(add_numbers, 5, 10, samples=3)

        assert metrics["samples"] == 3
        assert all(latency >= 0 for latency in metrics["all_samples_ms"])

    def test_sample_memory_single(self):
        """Test memory sampling (single snapshot)."""
        metrics = sample_memory()

        assert "ram_peak_gb" in metrics
        assert "ram_avg_gb" in metrics
        assert "vram_peak_gb" in metrics
        assert "vram_avg_gb" in metrics
        assert "ram_samples_count" in metrics

        assert metrics["ram_peak_gb"] > 0  # Should have some RAM usage
        assert metrics["ram_samples_count"] >= 1

    def test_sample_memory_duration(self):
        """Test memory sampling over duration."""
        # Sample for 0.1 seconds
        metrics = sample_memory(sample_interval=0.02, sample_duration=0.1)

        assert metrics["ram_samples_count"] >= 4  # At least 4 samples in 0.1s at 0.02s interval
        assert metrics["ram_avg_gb"] > 0
        assert metrics["ram_peak_gb"] >= metrics["ram_avg_gb"]

    def test_log_hardware_metadata(self):
        """Test hardware metadata collection."""
        metadata = log_hardware_metadata()

        # Check required fields
        assert "cpu" in metadata
        assert "ram" in metadata
        assert "gpu" in metadata
        assert "python" in metadata
        assert "os" in metadata
        assert "git" in metadata

        # CPU info
        assert "model" in metadata["cpu"]
        assert "cores_physical" in metadata["cpu"]
        assert "cores_logical" in metadata["cpu"]

        # RAM info
        assert "total_gb" in metadata["ram"]
        assert metadata["ram"]["total_gb"] > 0

        # Python info
        assert "version" in metadata["python"]

        # OS info
        assert "system" in metadata["os"]

    def test_format_hardware_summary(self):
        """Test hardware summary formatting."""
        metadata = log_hardware_metadata()
        summary = format_hardware_summary(metadata)

        assert isinstance(summary, str)
        assert "Hardware Configuration:" in summary
        assert "CPU:" in summary
        assert "RAM:" in summary
        assert "Python:" in summary
        assert "OS:" in summary

    def test_latency_percentiles_ordering(self):
        """Test that latency percentiles are ordered correctly."""

        def dummy_func():
            pass

        metrics = sample_latency(dummy_func, samples=100)

        # p50 <= p95 <= p99 <= max
        assert metrics["p50_ms"] <= metrics["p95_ms"]
        assert metrics["p95_ms"] <= metrics["p99_ms"]
        assert metrics["p99_ms"] <= metrics["max_ms"]
        assert metrics["min_ms"] <= metrics["p50_ms"]

    def test_sample_latency_error_handling(self):
        """Test that latency sampling propagates errors."""

        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            sample_latency(failing_function, samples=1)

    def test_memory_metrics_valid_ranges(self):
        """Test that memory metrics are in valid ranges."""
        metrics = sample_memory()

        # All GB values should be positive
        assert metrics["ram_peak_gb"] >= 0
        assert metrics["ram_avg_gb"] >= 0
        assert metrics["vram_peak_gb"] >= 0
        assert metrics["vram_avg_gb"] >= 0

        # Peak should be >= avg
        assert metrics["ram_peak_gb"] >= metrics["ram_avg_gb"]
        if metrics["vram_peak_gb"] > 0:  # Only if GPU available
            assert metrics["vram_peak_gb"] >= metrics["vram_avg_gb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
