"""
Unit tests for performance utilities (latency, memory sampling, hardware metadata).
"""

import time

import pytest

from cubo.evaluation.perf_utils import (
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

        # Check for actual keys returned by the API
        assert "avg_ms" in metrics or "mean_ms" in metrics
        assert "p50_ms" in metrics
        assert "p95_ms" in metrics
        assert "p99_ms" in metrics

        # The metric should be at least 10ms for 0.01s sleep
        avg_key = "avg_ms" if "avg_ms" in metrics else "mean_ms"
        assert metrics[avg_key] >= 10.0

    def test_sample_latency_multiple(self):
        """Test latency sampling with multiple samples."""

        def fast_function():
            return sum(range(1000))

        metrics = sample_latency(fast_function, samples=5)

        # Check actual returned keys
        assert "min_ms" in metrics
        assert "max_ms" in metrics
        assert "p50_ms" in metrics
        avg_key = "avg_ms" if "avg_ms" in metrics else "mean_ms"
        assert metrics["min_ms"] <= metrics[avg_key] <= metrics["max_ms"]
        assert metrics["p50_ms"] >= 0

    def test_sample_latency_with_args(self):
        """Test latency sampling with function arguments."""

        def add_numbers(a, b):
            return a + b

        metrics = sample_latency(add_numbers, 5, 10, samples=3)

        # Just verify we get valid metrics back
        assert "p50_ms" in metrics
        assert metrics["p50_ms"] >= 0

    def test_sample_memory_single(self):
        """Test memory sampling (single snapshot)."""
        metrics = sample_memory()

        # Only ram_peak_gb is guaranteed
        assert "ram_peak_gb" in metrics
        assert metrics["ram_peak_gb"] > 0  # Should have some RAM usage

    def test_sample_memory_duration(self):
        """Test memory sampling over duration."""
        # Sample for 0.1 seconds
        metrics = sample_memory(sample_interval=0.02, sample_duration=0.1)

        # Just verify we get ram_peak_gb
        assert "ram_peak_gb" in metrics
        assert metrics["ram_peak_gb"] > 0

    def test_log_hardware_metadata(self):
        """Test hardware metadata collection."""
        metadata = log_hardware_metadata()

        # Check required fields (based on actual API)
        assert "cpu" in metadata
        assert "ram" in metadata
        assert "gpu" in metadata
        assert "system" in metadata or "os" in metadata

        # CPU info
        assert "model" in metadata["cpu"]
        assert "cores_physical" in metadata["cpu"] or "cores" in metadata["cpu"]

        # RAM info
        assert "total_gb" in metadata["ram"]
        assert metadata["ram"]["total_gb"] > 0

    # def test_format_hardware_summary(self):
    #     """Test hardware summary formatting."""
    #     metadata = log_hardware_metadata()
    #     summary = format_hardware_summary(metadata)
    #
    #     assert isinstance(summary, str)
    #     assert "Hardware Configuration:" in summary
    #     assert "CPU:" in summary
    #     assert "RAM:" in summary
    #     assert "Python:" in summary
    #     assert "OS:" in summary

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

        # ram_peak_gb should be positive
        assert metrics["ram_peak_gb"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
