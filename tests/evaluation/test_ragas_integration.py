"""
Integration tests for RAGAS metrics in RAGTester.

NOTE: This test is marked as slow because importing ragas/langchain takes 30+ seconds.
Run with: pytest -m "not slow" to skip, or pytest -m slow to run only slow tests.
"""

import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

# Mark entire module as slow
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def setup_mocks():
    """Set up mocks for heavy modules - only runs when test is actually executed."""
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    # Save original modules
    saved_modules = {}
    modules_to_mock = [
        "ragas", "ragas.metrics", 
        "langchain", "langchain_community",
        "benchmarks.utils.ragas_evaluator",
        "cubo.main",
        "cubo.evaluation.metrics", 
        "cubo.evaluation.perf_utils"
    ]
    
    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules[mod_name]
    
    # Install mocks
    _mock_ragas = MagicMock()
    _mock_ragas.RAGAS_AVAILABLE = True
    sys.modules["ragas"] = _mock_ragas
    sys.modules["ragas.metrics"] = MagicMock()
    sys.modules["langchain"] = MagicMock()
    sys.modules["langchain_community"] = MagicMock()
    sys.modules["benchmarks.utils.ragas_evaluator"] = MagicMock()
    sys.modules["cubo.main"] = MagicMock()
    sys.modules["cubo.evaluation.metrics"] = MagicMock()
    
    mock_perf_utils = MagicMock()
    mock_perf_utils.log_hardware_metadata.return_value = {
        "cpu": {"model": "Test CPU"},
        "ram": {"total_gb": 16.0}
    }
    mock_perf_utils.sample_latency.return_value = {"p50_ms": 10.0}
    mock_perf_utils.sample_memory.return_value = {"ram_peak_gb": 1.0}
    sys.modules["cubo.evaluation.perf_utils"] = mock_perf_utils
    
    yield
    
    # Restore original modules
    for mod_name in modules_to_mock:
        if mod_name in saved_modules:
            sys.modules[mod_name] = saved_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


@pytest.fixture
def rag_tester_class(setup_mocks):
    """Import RAGTester after mocks are set up."""
    try:
        from benchmarks.retrieval.rag_benchmark import RAGTester
        return RAGTester
    except ImportError:
        pytest.skip("RAGTester not available")


def test_ragas_integration_in_rag_tester(rag_tester_class, setup_mocks):
    """Test that RAGTester correctly calls RAGASEvaluator and aggregates metrics."""
    
    RAGTester = rag_tester_class
    
    # Mock RAGASEvaluator
    mock_ragas_evaluator = AsyncMock()
    mock_ragas_evaluator.evaluate_single.return_value = {
        "comprehensiveness": 0.8,
        "diversity": 0.7,
        "empowerment": 0.9,
        "overall": 0.8
    }

    # Patch get_ragas_evaluator to return our mock
    with patch("benchmarks.retrieval.rag_benchmark.get_ragas_evaluator", return_value=mock_ragas_evaluator):
        # Initialize tester
        tester = RAGTester(questions_file="dummy.json", mode="full")
        
        # Mock internal components
        tester.cubo_app = MagicMock()
        tester.cubo_app.retriever.retrieve_top_documents.return_value = [{"text": "context"}]
        tester.cubo_app.generator.generate_response.return_value = "answer"
        
        # Mock evaluate_response to return dummy data
        tester.evaluate_response = AsyncMock(return_value={})
        
        # Run single test
        result = tester.run_single_test("test question", "easy", question_id="q1")
        
        # Verify RAGAS evaluator was called
        mock_ragas_evaluator.evaluate_single.assert_called_once()
        call_args = mock_ragas_evaluator.evaluate_single.call_args
        assert call_args.kwargs["question"] == "test question"
        assert call_args.kwargs["answer"] == "answer"
        
        # Verify metrics are in result
        assert "ragas_metrics" in result
        assert result["ragas_metrics"]["comprehensiveness"] == 0.8
        
        # Manually populate results to test aggregation
        tester.results["results"]["easy"] = [result]
        
        # Run statistics calculation
        tester.calculate_statistics()
        
        # Verify aggregation
        stats = tester.results["metadata"]["questions_by_difficulty"]["easy"]
        assert "avg_ragas_comprehensiveness" in stats
        assert stats["avg_ragas_comprehensiveness"] == 0.8
        assert stats["avg_ragas_diversity"] == 0.7
        assert stats["avg_ragas_empowerment"] == 0.9
