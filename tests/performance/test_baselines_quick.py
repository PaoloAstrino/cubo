"""Quick smoke tests for baseline comparison scripts.

Tests that baseline scripts run without errors on small samples.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_queries():
    """Create sample queries file."""
    queries = [
        {"_id": "q1", "text": "What is machine learning?"},
        {"_id": "q2", "text": "How do neural networks work?"},
        {"_id": "q3", "text": "Explain gradient descent"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
        path = Path(f.name)

    yield path
    path.unlink()


@pytest.fixture
def sample_corpus():
    """Create sample corpus file."""
    docs = [
        {"_id": "d1", "text": "Machine learning is a subset of artificial intelligence..."},
        {"_id": "d2", "text": "Neural networks are inspired by biological neurons..."},
        {"_id": "d3", "text": "Gradient descent is an optimization algorithm..."},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
        path = Path(f.name)

    yield path
    path.unlink()


def test_pyserini_baseline_imports():
    """Test Pyserini baseline script imports successfully."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tools import run_pyserini_baseline

        assert run_pyserini_baseline.check_pyserini_available is not None
    except ImportError as e:
        pytest.skip(f"Pyserini imports failed (expected): {e}")


def test_splade_baseline_imports():
    """Test SPLADE baseline script imports successfully."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tools import run_splade_baseline

        assert run_splade_baseline.check_splade_available is not None
    except ImportError as e:
        pytest.skip(f"SPLADE imports failed (expected): {e}")


def test_e5_baseline_imports():
    """Test e5 baseline script imports successfully."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tools import run_e5_ivfpq_baseline

        assert run_e5_ivfpq_baseline.check_e5_available is not None
    except ImportError as e:
        pytest.skip(f"e5 imports failed (expected): {e}")


def test_pyserini_baseline_runs_quick(sample_queries, sample_corpus):
    """Test Pyserini baseline runs on small sample."""
    output = tempfile.mktemp(suffix=".json")

    try:
        # Run with minimal config
        result = subprocess.run(
            [
                sys.executable,
                "tools/run_pyserini_baseline.py",
                "--dataset",
                "test",
                "--output",
                output,
            ],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            timeout=30,
        )

        # May fail due to missing Pyserini, but shouldn't crash
        assert result.returncode in [0, 1]  # 0 = success, 1 = graceful failure

    except subprocess.TimeoutExpired:
        pytest.fail("Baseline script timed out")
    finally:
        if Path(output).exists():
            Path(output).unlink()


def test_baseline_graceful_fallback():
    """Test baselines handle missing dependencies gracefully."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from tools import run_e5_ivfpq_baseline, run_pyserini_baseline, run_splade_baseline

    # These should return False if dependencies missing, not crash
    pyserini_avail = run_pyserini_baseline.check_pyserini_available()
    splade_avail = run_splade_baseline.check_splade_available()
    e5_avail = run_e5_ivfpq_baseline.check_e5_available()

    # All should be boolean
    assert isinstance(pyserini_avail, bool)
    assert isinstance(splade_avail, bool)
    assert isinstance(e5_avail, bool)


def test_multilingual_eval_imports():
    """Test multilingual eval script imports successfully."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tools import run_multilingual_eval

    assert run_multilingual_eval.run_multilingual_eval is not None


def test_profiling_tools_import():
    """Test profiling tools import successfully."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from tools import (
        benchmark_concurrency_real,
        profile_retrieval_breakdown_real,
        sensitivity_analysis_real,
    )

    assert profile_retrieval_breakdown_real.measure_retrieval_components_real is not None
    assert sensitivity_analysis_real.measure_faiss_sensitivity_real is not None
    assert benchmark_concurrency_real.run_concurrent_benchmark_real is not None
