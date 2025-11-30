"""
Unit tests for perf utilities.
Copied from original test_perf_utils.py.
"""

from cubo.evaluation.perf_utils import sample_latency, sample_memory


def test_sample_latency_simple():
    def fast_fn(x):
        return x

    res = sample_latency(fast_fn, "test", samples=3)
    assert "p50_ms" in res and res["p50_ms"] >= 0


def test_sample_memory_snapshot():
    mem = sample_memory(sample_duration=0.01, sample_interval=0.01)
    assert "ram_peak_gb" in mem
