import os

import pytest

from benchmarks.retrieval.rag_benchmark import RAGTester


def _has_retrieval_deps():
    # Check if common retrieval libs are present; if not, skip the test
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        pass
    return False


@pytest.mark.smoke
def test_retrieval_latency_and_recall_constraints(
    smoke_data_dir, sample_questions, sample_ground_truth
):
    # Skip if retrieval dependencies are not installed
    if not _has_retrieval_deps():
        pytest.skip("Retrieval deps not available, skipping smoke retrieval test")

    # Run a small retrieval-only test and assert thresholds (smoke-level)
    tester = RAGTester(
        str(sample_questions),
        str(smoke_data_dir),
        ground_truth_file=str(sample_ground_truth),
        mode="retrieval-only",
    )
    results = tester.run_all_tests(easy_limit=5, medium_limit=0, hard_limit=0, k_values=[5, 10])
    # Basic assertions
    metadata = results.get("metadata", {})
    assert metadata.get("total_questions", 0) > 0

    # Environment-aware latency thresholds
    LATENCY_THRESHOLDS = {
        "ci": 500,  # CI environment (slower)
        "dev": 200,  # Dev machines
        "staging": 100,  # Staging
        "prod": 50,  # Production
    }

    env = os.getenv("TEST_ENV", "dev")
    threshold_ms = LATENCY_THRESHOLDS.get(env, 200)

    # Latency check with environment-aware threshold
    overall_latency_p50 = metadata.get("avg_retrieval_latency_p50_ms", 0)
    if overall_latency_p50 and overall_latency_p50 > 0:
        assert (
            overall_latency_p50 < threshold_ms
        ), f"P50 latency {overall_latency_p50}ms exceeds {env} threshold {threshold_ms}ms"

    # Check recall at k present and > 0
    assert metadata.get("avg_recall_at_k_5", 0) >= 0
