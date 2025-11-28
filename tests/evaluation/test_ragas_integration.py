"""
Integration tests for RAGAS metrics in RAGTester.
"""

import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

# Add project root to path to import from benchmarks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock dependencies before importing run_rag_tests
sys.modules["src.cubo.main"] = MagicMock()
sys.modules["src.cubo.evaluation.metrics"] = MagicMock()
sys.modules["src.cubo.evaluation.perf_utils"] = MagicMock()

# Configure perf_utils mock to return dict for hardware metadata
# This prevents TypeError when f-string tries to format MagicMock
sys.modules["src.cubo.evaluation.perf_utils"].log_hardware_metadata.return_value = {
    "cpu": {"model": "Test CPU"},
    "ram": {"total_gb": 16.0}
}
sys.modules["src.cubo.evaluation.perf_utils"].sample_latency.return_value = {"p50_ms": 10.0}
sys.modules["src.cubo.evaluation.perf_utils"].sample_memory.return_value = {"ram_peak_gb": 1.0}

try:
    from benchmarks.retrieval.rag_benchmark import RAGTester
except ImportError:
    RAGTester = None


@pytest.mark.skipif(RAGTester is None, reason="RAGTester not available")
def test_ragas_integration_in_rag_tester():
    """Test that RAGTester correctly calls RAGASEvaluator and aggregates metrics."""
    
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
