"""
Tests for ServiceManager
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from src.service_manager import ServiceManager, get_service_manager, shutdown_service_manager
from src.health_monitor import HealthStatus


class TestServiceManager:
    """Test cases for ServiceManager class."""

    def test_initialization(self):
        """Test ServiceManager initialization."""
        sm = ServiceManager(max_workers=2)

        assert sm.thread_manager is not None
        assert sm.error_recovery is not None
        assert sm.health_monitor is not None

        # Check that component health checks are registered
        assert 'thread_pool' in sm.health_monitor.health_checks
        assert 'error_recovery' in sm.health_monitor.health_checks

        sm.shutdown(wait=True)

    def test_execute_async_with_retry(self):
        """Test async execution with retry."""
        sm = ServiceManager(max_workers=2)

        call_count = [0]

        def success_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"

        future = sm.execute_async('document_processing', success_operation, with_retry=True)
        result = future.result(timeout=10)

        assert result == "success"
        assert call_count[0] == 2  # One failure, one success

        sm.shutdown(wait=True)

    def test_execute_async_without_retry(self):
        """Test async execution without retry."""
        sm = ServiceManager(max_workers=2)

        def failing_operation():
            raise ValueError("Always fails")

        future = sm.execute_async('document_processing', failing_operation, with_retry=False)

        with pytest.raises(ValueError, match="Always fails"):
            future.result(timeout=5)

        sm.shutdown(wait=True)

    def test_execute_sync_success(self):
        """Test synchronous execution success."""
        sm = ServiceManager(max_workers=2)

        def success_operation():
            return "sync_success"

        result = sm.execute_sync('document_processing', success_operation)
        assert result == "sync_success"

        sm.shutdown(wait=True)

    def test_execute_sync_with_fallback(self):
        """Test synchronous execution with fallback strategy."""
        sm = ServiceManager(max_workers=2)

        def failing_llm_operation():
            raise RuntimeError("LLM failed")

        result = sm.execute_sync('llm_generation', failing_llm_operation)
        expected_fallback = "I apologize, but I'm unable to generate a response at this time. Please try again."
        assert result == expected_fallback

        sm.shutdown(wait=True)

    def test_get_system_status(self):
        """Test getting comprehensive system status."""
        sm = ServiceManager(max_workers=2)

        status = sm.get_system_status()

        assert 'threads' in status
        assert 'health' in status
        assert 'errors' in status
        assert 'timestamp' in status

        # Check thread status
        assert 'active_tasks' in status['threads']
        assert 'max_workers' in status['threads']
        assert 'shutdown' in status['threads']

        # Check health status
        assert 'overall_status' in status['health']
        assert 'checks' in status['health']

        # Check error status
        assert isinstance(status['errors'], dict)

        sm.shutdown(wait=True)

    def test_wait_for_completion(self):
        """Test waiting for operation completion."""
        sm = ServiceManager(max_workers=2)

        def quick_task():
            return "done"

        def slow_task():
            time.sleep(0.5)
            return "slow_done"

        # Submit tasks
        sm.execute_async('test_op', quick_task)
        sm.execute_async('test_op', slow_task)

        # Wait for completion
        success = sm.wait_for_completion(timeout=2.0)
        assert success

        sm.shutdown(wait=True)

    def test_wait_for_completion_timeout(self):
        """Test wait_for_completion with timeout."""
        sm = ServiceManager(max_workers=2)

        def very_slow_task():
            time.sleep(2)
            return "very_slow"

        sm.execute_async('test_op', very_slow_task)

        # Should timeout before task completes
        success = sm.wait_for_completion(timeout=0.5)
        assert not success

        sm.shutdown(wait=True)

    def test_shutdown(self):
        """Test service manager shutdown."""
        sm = ServiceManager(max_workers=2)

        def simple_task():
            return "ok"

        # Submit a task
        future = sm.execute_async('test_op', simple_task)

        # Shutdown
        sm.shutdown(wait=True)

        # Task should complete
        result = future.result(timeout=1)
        assert result == "ok"

    def test_operation_context_manager(self):
        """Test operation context manager."""
        sm = ServiceManager(max_workers=2)

        def context_task():
            return "context_result"

        with sm.operation_context(timeout=5):
            future = sm.execute_async('test_op', context_task)
            result = future.result()
            assert result == "context_result"

        # Should cleanup automatically
        assert sm.thread_manager.get_active_count() == 0

        sm.shutdown(wait=True)

    def test_thread_pool_health_check(self):
        """Test thread pool health monitoring."""
        sm = ServiceManager(max_workers=2)

        # Initially healthy
        health_result = sm._check_thread_pool_health()
        assert health_result['status'] == HealthStatus.HEALTHY.value
        assert health_result['active_tasks'] == 0
        assert health_result['max_workers'] == 2

        # Submit some tasks to make it active
        def active_task():
            time.sleep(0.1)
            return "active"

        futures = []
        for i in range(3):  # More than max_workers
            future = sm.execute_async('test_op', active_task)
            futures.append(future)

        # Give it a moment to start
        time.sleep(0.05)

        # Check health while active
        health_result = sm._check_thread_pool_health()
        assert health_result['active_tasks'] > 0

        # Wait for completion
        for future in futures:
            future.result(timeout=1)

        sm.shutdown(wait=True)

    def test_error_recovery_health_check(self):
        """Test error recovery health monitoring."""
        sm = ServiceManager(max_workers=2)

        # Initially healthy
        health_result = sm._check_error_recovery_health()
        assert health_result['status'] == HealthStatus.HEALTHY.value
        assert len(health_result['critical_operations']) == 0
        assert len(health_result['warning_operations']) == 0

        # Simulate some errors
        def failing_op():
            raise ValueError("Test failure")

        for _ in range(5):  # Create high failure rate
            try:
                sm.execute_sync('test_failing', failing_op)
            except ValueError:
                pass

        # Check health after failures
        health_result = sm._check_error_recovery_health()
        assert health_result['status'] in [HealthStatus.WARNING.value, HealthStatus.CRITICAL.value]
        assert 'test_failing' in health_result['critical_operations'] or 'test_failing' in health_result['warning_operations']

        sm.shutdown(wait=True)

    def test_health_alert_handling(self):
        """Test health alert callback handling."""
        sm = ServiceManager(max_workers=2)

        alerts_received = []

        def alert_callback(check_name, status, details):
            alerts_received.append((check_name, status.value))

        # Add our test callback
        sm.health_monitor.add_alert_callback(alert_callback)

        # Trigger a health alert by creating a failing health check
        def failing_health_check():
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': 'Test critical alert',
                'timestamp': time.time()
            }

        sm.health_monitor.add_health_check('test_alert', failing_health_check)

        # Run the check to trigger alert
        sm.health_monitor.perform_health_check('test_alert')

        assert len(alerts_received) > 0
        assert any(alert[0] == 'test_alert' for alert in alerts_received)

        sm.shutdown(wait=True)

    def test_convenience_methods(self):
        """Test convenience methods for common operations."""
        sm = ServiceManager(max_workers=2)

        # Test document processing
        def doc_processor(filepath):
            return f"Processed {filepath}"

        future = sm.process_document_async("test.txt", doc_processor)
        result = future.result(timeout=5)
        assert result == "Processed test.txt"

        # Test embedding generation
        def embedding_generator(text):
            return f"Embedding for {text}"

        future = sm.generate_embedding_async("test text", embedding_generator)
        result = future.result(timeout=5)
        assert result == "Embedding for test text"

        # Test database query
        def db_query():
            return "query_result"

        future = sm.query_database_async(db_query)
        result = future.result(timeout=5)
        assert result == "query_result"

        # Test response generation
        def response_generator(query, context):
            return f"Response to {query} with {context}"

        future = sm.generate_response_async("test query", "test context", response_generator)
        result = future.result(timeout=5)
        assert result == "Response to test query with test context"

        sm.shutdown(wait=True)

    def test_global_service_manager(self):
        """Test global service manager instance."""
        # Shutdown any existing instance
        shutdown_service_manager()

        # Get new instance
        sm1 = get_service_manager()
        assert sm1 is not None

        # Get same instance again
        sm2 = get_service_manager()
        assert sm1 is sm2

        # Shutdown
        shutdown_service_manager()

        # Get new instance after shutdown
        sm3 = get_service_manager()
        assert sm3 is not sm1  # Should be different instance

        sm3.shutdown(wait=True)

    def test_multiple_operations_concurrent(self):
        """Test handling multiple concurrent operations."""
        sm = ServiceManager(max_workers=4)

        results = []
        lock = threading.Lock()

        def concurrent_operation(task_id):
            time.sleep(0.1)  # Simulate work
            with lock:
                results.append(task_id)
            return task_id

        # Submit multiple operations
        futures = []
        for i in range(8):  # More than max_workers
            future = sm.execute_async('concurrent_test', concurrent_operation, i)
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            future.result(timeout=5)

        assert len(results) == 8
        assert set(results) == set(range(8))

        sm.shutdown(wait=True)

    def test_error_propagation(self):
        """Test that errors are properly propagated."""
        sm = ServiceManager(max_workers=2)

        def operation_with_error():
            raise ValueError("Test error")

        # Async error
        future = sm.execute_async('test_error', operation_with_error)
        with pytest.raises(ValueError, match="Test error"):
            future.result(timeout=5)

        # Sync error
        with pytest.raises(ValueError, match="Test error"):
            sm.execute_sync('test_error', operation_with_error)

        sm.shutdown(wait=True)

    def test_component_integration(self):
        """Test integration between all components."""
        sm = ServiceManager(max_workers=2)

        # Test that thread manager, error recovery, and health monitor work together
        def test_operation():
            return "integrated_test"

        # Execute operation
        future = sm.execute_async('integration_test', test_operation)
        result = future.result(timeout=5)
        assert result == "integrated_test"

        # Check system status includes all components
        status = sm.get_system_status()
        assert 'threads' in status
        assert 'health' in status
        assert 'errors' in status

        # Check health includes component checks
        health_checks = status['health']['checks']
        assert 'thread_pool' in health_checks
        assert 'error_recovery' in health_checks

        sm.shutdown(wait=True)