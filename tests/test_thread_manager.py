"""
Tests for ThreadManager
"""

import pytest
import time
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from unittest.mock import patch, MagicMock

from src.cubo.workers.thread_manager import ThreadManager


class TestThreadManager:
    """Test cases for ThreadManager class."""

    def test_initialization(self):
        """Test ThreadManager initialization."""
        tm = ThreadManager(max_workers=2, thread_name_prefix="test")
        assert tm.executor._max_workers == 2
        assert not tm._shutdown
        assert len(tm.active_futures) == 0
        tm.shutdown(wait=True)

    def test_submit_task_basic(self):
        """Test basic task submission."""
        tm = ThreadManager(max_workers=2)

        def simple_task():
            return 42

        future = tm.submit_task(simple_task)
        result = future.result(timeout=5)
        assert result == 42
        assert tm.get_active_count() == 0  # Should be cleaned up

        tm.shutdown(wait=True)

    def test_submit_task_with_args_kwargs(self):
        """Test task submission with arguments."""
        tm = ThreadManager(max_workers=2)

        def task_with_args(x, y, multiplier=1):
            return (x + y) * multiplier

        future = tm.submit_task(task_with_args, 3, 4, multiplier=2)
        result = future.result(timeout=5)
        assert result == 14  # (3+4)*2

        tm.shutdown(wait=True)

    def test_submit_task_timeout(self):
        """Test task timeout functionality."""
        tm = ThreadManager(max_workers=2)

        def slow_task():
            time.sleep(2)
            return "done"

        future = tm.submit_task(slow_task, timeout=0.5)

        with pytest.raises(FutureTimeoutError):
            future.result(timeout=1)

        tm.shutdown(wait=True)

    def test_submit_task_with_retry_success(self):
        """Test task retry on failure."""
        tm = ThreadManager(max_workers=2)

        call_count = [0]

        def failing_task():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        future = tm.submit_task_with_retry(failing_task, max_retries=3, retry_delay=0.1)
        result = future.result(timeout=5)
        assert result == "success"
        assert call_count[0] == 3

        tm.shutdown(wait=True)

    def test_submit_task_with_retry_exhaustion(self):
        """Test task retry exhaustion."""
        tm = ThreadManager(max_workers=2)

        def always_failing_task():
            raise ValueError("Always fails")

        future = tm.submit_task_with_retry(always_failing_task, max_retries=2, retry_delay=0.1)

        with pytest.raises(ValueError, match="Always fails"):
            future.result(timeout=5)

        tm.shutdown(wait=True)

    def test_wait_for_all(self):
        """Test waiting for all tasks to complete."""
        tm = ThreadManager(max_workers=3)

        results = []

        def task(delay, result):
            time.sleep(delay)
            results.append(result)
            return result

        # Submit multiple tasks
        futures = []
        for i in range(3):
            future = tm.submit_task(task, 0.1 * (i + 1), f"task_{i}")
            futures.append(future)

        # Wait for all
        success = tm.wait_for_all(timeout=2.0)
        assert success
        assert len(results) == 3
        assert tm.get_active_count() == 0

        tm.shutdown(wait=True)

    def test_wait_for_all_timeout(self):
        """Test wait_for_all with timeout."""
        tm = ThreadManager(max_workers=2)

        def slow_task():
            time.sleep(1)
            return "done"

        tm.submit_task(slow_task)
        tm.submit_task(slow_task)

        # Should timeout before tasks complete
        success = tm.wait_for_all(timeout=0.5)
        assert not success
        assert tm.get_active_count() > 0

        tm.shutdown(wait=True)

    def test_get_active_count(self):
        """Test active task counting."""
        tm = ThreadManager(max_workers=2)

        def quick_task():
            return "done"

        def slow_task():
            time.sleep(0.5)
            return "slow_done"

        # Submit tasks
        tm.submit_task(quick_task)
        tm.submit_task(slow_task)

        # Should have active tasks initially
        assert tm.get_active_count() >= 1

        # Wait for completion
        time.sleep(0.6)
        assert tm.get_active_count() == 0

        tm.shutdown(wait=True)

    def test_get_status(self):
        """Test status reporting."""
        tm = ThreadManager(max_workers=3)

        status = tm.get_status()
        assert status["max_workers"] == 3
        assert status["shutdown"] is False
        assert "active_tasks" in status

        tm.shutdown(wait=True)

        status = tm.get_status()
        assert status["shutdown"] is True

    def test_shutdown_behavior(self):
        """Test shutdown functionality."""
        tm = ThreadManager(max_workers=2)

        def slow_task():
            time.sleep(0.5)
            return "done"

        # Submit task before shutdown
        future = tm.submit_task(slow_task)

        # Shutdown with wait
        tm.shutdown(wait=True)

        # Should complete successfully
        result = future.result(timeout=1)
        assert result == "done"

        # Should not accept new tasks
        with pytest.raises(RuntimeError, match="ThreadManager is shutting down"):
            tm.submit_task(lambda: None)

    def test_shutdown_without_wait(self):
        """Test shutdown without waiting."""
        tm = ThreadManager(max_workers=2)

        def slow_task():
            time.sleep(1)
            return "done"

        tm.submit_task(slow_task)

        # Shutdown without waiting
        start_time = time.time()
        tm.shutdown(wait=False)
        shutdown_time = time.time() - start_time

        # Should shutdown quickly without waiting for task
        assert shutdown_time < 0.5

    def test_task_context_manager(self):
        """Test task context manager."""
        tm = ThreadManager(max_workers=2)

        def simple_task():
            return "context_test"

        with tm.task_context(timeout=5):
            future = tm.submit_task(simple_task)
            result = future.result()
            assert result == "context_test"

        # Should cleanup automatically
        assert tm.get_active_count() == 0

        tm.shutdown(wait=True)

    def test_exception_handling(self):
        """Test exception handling in tasks."""
        tm = ThreadManager(max_workers=2)

        def failing_task():
            raise ValueError("Test exception")

        future = tm.submit_task(failing_task)

        with pytest.raises(ValueError, match="Test exception"):
            future.result(timeout=5)

        # Should still cleanup
        assert tm.get_active_count() == 0

        tm.shutdown(wait=True)

    def test_multiple_simultaneous_tasks(self):
        """Test handling multiple simultaneous tasks."""
        tm = ThreadManager(max_workers=4)

        results = []
        lock = threading.Lock()

        def concurrent_task(task_id):
            time.sleep(0.1)  # Simulate work
            with lock:
                results.append(task_id)
            return task_id

        # Submit many tasks
        futures = []
        for i in range(10):
            future = tm.submit_task(concurrent_task, i)
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            future.result(timeout=5)

        assert len(results) == 10
        assert set(results) == set(range(10))
        assert tm.get_active_count() == 0

        tm.shutdown(wait=True)