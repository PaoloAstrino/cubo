"""
Tests for ErrorRecoveryManager
"""

import time

import pytest

from src.cubo.utils.error_recovery import ErrorRecoveryManager, RecoveryStrategy


class TestErrorRecoveryManager:
    """Test cases for ErrorRecoveryManager class."""

    def test_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        erm = ErrorRecoveryManager()

        # Check default configurations
        assert "document_processing" in erm.recovery_configs
        assert "llm_generation" in erm.recovery_configs
        assert erm.recovery_configs["document_processing"]["strategy"] == RecoveryStrategy.RETRY
        assert erm.recovery_configs["llm_generation"]["strategy"] == RecoveryStrategy.FALLBACK

        # Check initial state
        assert len(erm.error_counts) == 0
        assert len(erm.last_errors) == 0

    def test_execute_with_recovery_success(self):
        """Test successful operation execution."""
        erm = ErrorRecoveryManager()

        def successful_operation():
            return "success"

        result = erm.execute_with_recovery("document_processing", successful_operation)
        assert result == "success"

        # Check error tracking
        status = erm.get_health_status()
        assert "document_processing" in status
        assert status["document_processing"]["healthy"] is True
        assert status["document_processing"]["total_attempts"] == 1
        assert status["document_processing"]["failure_rate"] == 0.0

    def test_execute_with_recovery_retry_success(self):
        """Test operation that succeeds after retries."""
        erm = ErrorRecoveryManager()

        call_count = [0]

        def eventually_successful_operation():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = erm.execute_with_recovery("document_processing", eventually_successful_operation)
        assert result == "success"
        assert call_count[0] == 3

        # Check error tracking
        status = erm.get_health_status()
        assert status["document_processing"]["total_attempts"] == 3
        assert status["document_processing"]["failure_rate"] == 2 / 3

    def test_execute_with_recovery_retry_exhaustion(self):
        """Test operation that fails after all retries."""
        erm = ErrorRecoveryManager()

        def always_failing_operation():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            erm.execute_with_recovery("document_processing", always_failing_operation)

        # Check error tracking
        status = erm.get_health_status()
        assert status["document_processing"]["healthy"] is False
        assert status["document_processing"]["failure_rate"] == 1.0
        assert status["document_processing"]["recent_failure"] is True

    def test_execute_with_recovery_fallback(self):
        """Test fallback strategy for LLM generation."""
        erm = ErrorRecoveryManager()

        def failing_llm_operation():
            raise RuntimeError("LLM service unavailable")

        result = erm.execute_with_recovery("llm_generation", failing_llm_operation)
        expected_fallback = (
            "I apologize, but I'm unable to generate a response at this time. Please try again."
        )
        assert result == expected_fallback

    def test_execute_with_recovery_skip(self):
        """Test skip strategy."""
        erm = ErrorRecoveryManager()

        # Add skip configuration
        erm.add_recovery_config(
            "test_skip",
            {
                "strategy": RecoveryStrategy.SKIP,
                "max_retries": 1,
                "retry_delay": 0.1,
                "timeout": 1.0,
            },
        )

        def failing_operation():
            raise ValueError("Test error")

        result = erm.execute_with_recovery("test_skip", failing_operation)
        assert result is None

    def test_execute_with_recovery_timeout(self):
        """Test operation timeout."""
        erm = ErrorRecoveryManager()

        def slow_operation():
            time.sleep(2)
            return "done"

        # Use a very short timeout
        erm.add_recovery_config(
            "test_timeout",
            {
                "strategy": RecoveryStrategy.RETRY,
                "max_retries": 0,
                "retry_delay": 0.1,
                "timeout": 0.5,
            },
        )

        with pytest.raises(TimeoutError, match="timed out after 0.5 seconds"):
            erm.execute_with_recovery("test_timeout", slow_operation)

    def test_execute_with_recovery_unknown_operation(self):
        """Test execution with unknown operation type."""
        erm = ErrorRecoveryManager()

        def simple_operation():
            return "success"

        # Should use default config
        result = erm.execute_with_recovery("unknown_operation", simple_operation)
        assert result == "success"

    def test_get_health_status_comprehensive(self):
        """Test comprehensive health status reporting."""
        erm = ErrorRecoveryManager()

        # Simulate various scenarios
        def success_op():
            return "ok"

        def fail_op():
            raise ValueError("fail")

        # Successful operation
        erm.execute_with_recovery("test_op1", success_op)

        # Failed operation
        try:
            erm.execute_with_recovery("test_op2", fail_op)
        except ValueError:
            pass

        # Mixed operation (succeeds after retry) - should have 50% failure rate
        call_count = [0]

        def retry_success_op():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("temp fail")
            return "ok"

        erm.execute_with_recovery("test_op3", retry_success_op)

        status = erm.get_health_status()

        assert status["test_op1"]["healthy"] is True
        assert status["test_op1"]["failure_rate"] == 0.0

        assert status["test_op2"]["healthy"] is False
        assert status["test_op2"]["failure_rate"] == 1.0

        assert status["test_op3"]["healthy"] is False  # 50% failure rate makes it unhealthy
        assert status["test_op3"]["failure_rate"] == 0.5  # 1 failure out of 2 attempts

    def test_reset_error_counts_specific(self):
        """Test resetting error counts for specific operation."""
        erm = ErrorRecoveryManager()

        def fail_op():
            raise ValueError("fail")

        try:
            erm.execute_with_recovery("test_op", fail_op)
        except ValueError:
            pass

        # Verify error is recorded
        status = erm.get_health_status()
        assert "test_op" in status

        # Reset specific operation
        erm.reset_error_counts("test_op")

        # Verify it's gone
        status = erm.get_health_status()
        assert "test_op" not in status

    def test_reset_error_counts_all(self):
        """Test resetting all error counts."""
        erm = ErrorRecoveryManager()

        def fail_op():
            raise ValueError("fail")

        try:
            erm.execute_with_recovery("test_op1", fail_op)
        except ValueError:
            pass

        try:
            erm.execute_with_recovery("test_op2", fail_op)
        except ValueError:
            pass

        # Verify errors are recorded
        status = erm.get_health_status()
        assert len(status) >= 2

        # Reset all
        erm.reset_error_counts()

        # Verify all are gone
        status = erm.get_health_status()
        assert len(status) == 0

    def test_add_recovery_config(self):
        """Test adding custom recovery configuration."""
        erm = ErrorRecoveryManager()

        custom_config = {
            "strategy": RecoveryStrategy.RETRY,
            "max_retries": 5,
            "retry_delay": 2.0,
            "timeout": 60.0,
        }

        erm.add_recovery_config("custom_operation", custom_config)

        assert "custom_operation" in erm.recovery_configs
        assert erm.recovery_configs["custom_operation"] == custom_config

        # Test using the custom config
        def success_op():
            return "custom_success"

        result = erm.execute_with_recovery("custom_operation", success_op)
        assert result == "custom_success"

    def test_exponential_backoff(self):
        """Test exponential backoff in retry delays."""
        erm = ErrorRecoveryManager()

        call_times = []

        def failing_operation():
            call_times.append(time.time())
            raise ValueError("fail")

        start_time = time.time()

        try:
            erm.execute_with_recovery("document_processing", failing_operation)
        except ValueError:
            pass

        # Should have 4 calls (initial + 3 retries)
        assert len(call_times) == 4

        # Check delays between calls (should increase exponentially)
        delays = []
        for i in range(1, len(call_times)):
            delays.append(call_times[i] - call_times[i - 1])

        # First retry delay should be ~1.0s, second ~1.5s, third ~2.25s
        assert 0.9 <= delays[0] <= 1.1  # First retry
        assert 1.4 <= delays[1] <= 1.6  # Second retry
        assert 2.2 <= delays[2] <= 2.3  # Third retry

    def test_error_tracking_persistence(self):
        """Test that error tracking persists across multiple operations."""
        erm = ErrorRecoveryManager()

        def fail_op():
            raise ValueError("fail")

        def success_op():
            return "ok"

        # Multiple failures
        for _ in range(3):
            try:
                erm.execute_with_recovery("persistent_test", fail_op)
            except ValueError:
                pass

        # One success
        erm.execute_with_recovery("persistent_test", success_op)

        status = erm.get_health_status()["persistent_test"]
        assert status["total_attempts"] == 7  # 3 failures (each with 2 attempts) + 1 success
        assert status["failure_rate"] == 6 / 7

    def test_recent_failure_detection(self):
        """Test recent failure detection in health status."""
        erm = ErrorRecoveryManager()

        def fail_op():
            raise ValueError("fail")

        # Simulate old failure
        try:
            erm.execute_with_recovery("recent_test", fail_op)
        except ValueError:
            pass

        # Manually set last failure to old time
        erm.error_counts["recent_test"]["last_failure"] = time.time() - 400  # 400 seconds ago

        status = erm.get_health_status()["recent_test"]
        assert status["recent_failure"] is False  # Not recent

        # Now simulate recent failure
        try:
            erm.execute_with_recovery("recent_test", fail_op)
        except ValueError:
            pass

        status = erm.get_health_status()["recent_test"]
        assert status["recent_failure"] is True  # Recent
