"""
Tests for the BackgroundTaskManager.

These tests verify the background task management system:
- Task submission and execution
- Progress tracking
- Cancellation
- Timeout handling
- Task lifecycle management
"""

import time

import pytest

from cubo.workers.background_task_manager import (
    BackgroundTask,
    BackgroundTaskManager,
    TaskProgress,
    TaskResult,
    TaskState,
)


@pytest.fixture
def task_manager():
    """Create a fresh task manager for each test."""
    # Create a non-singleton instance for testing
    manager = object.__new__(BackgroundTaskManager)
    manager._initialized = False
    manager.__init__(max_workers=2, max_task_age_hours=1)
    yield manager
    manager.shutdown(wait=False)


class TestTaskProgress:
    """Tests for TaskProgress dataclass."""

    def test_percentage_calculation(self):
        """Test progress percentage calculation."""
        progress = TaskProgress(current=50, total=100)
        assert progress.percentage == 50.0

    def test_percentage_zero_total(self):
        """Test percentage with zero total."""
        progress = TaskProgress(current=10, total=0)
        assert progress.percentage == 0.0

    def test_percentage_over_100_capped(self):
        """Test percentage is capped at 100."""
        progress = TaskProgress(current=150, total=100)
        assert progress.percentage == 100.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = TaskProgress(current=25, total=100, message="Working", phase="Phase 1")
        result = progress.to_dict()

        assert result["current"] == 25
        assert result["total"] == 100
        assert result["percentage"] == 25.0
        assert result["message"] == "Working"
        assert result["phase"] == "Phase 1"


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = TaskResult(success=True, value={"data": "test"}, execution_time_ms=100.0)

        assert result.success
        assert result.value == {"data": "test"}
        assert result.error is None

    def test_failure_result(self):
        """Test failure result."""
        result = TaskResult(
            success=False,
            error="Something went wrong",
            error_type="ValueError",
            execution_time_ms=50.0,
        )

        assert not result.success
        assert result.value is None
        assert result.error == "Something went wrong"
        assert result.error_type == "ValueError"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TaskResult(success=True, value=42, execution_time_ms=123.45)
        d = result.to_dict()

        assert d["success"] is True
        assert d["value"] == 42
        assert d["execution_time_ms"] == 123.45


class TestBackgroundTask:
    """Tests for BackgroundTask dataclass."""

    def test_is_complete_states(self):
        """Test is_complete property for various states."""
        from datetime import datetime

        task = BackgroundTask(
            task_id="test-123",
            name="Test Task",
            state=TaskState.PENDING,
            created_at=datetime.utcnow(),
        )

        assert not task.is_complete

        task.state = TaskState.RUNNING
        assert not task.is_complete

        task.state = TaskState.COMPLETED
        assert task.is_complete

        task.state = TaskState.FAILED
        assert task.is_complete

        task.state = TaskState.CANCELLED
        assert task.is_complete

    def test_duration_calculation(self):
        """Test duration calculation."""
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        task = BackgroundTask(
            task_id="test-123",
            name="Test Task",
            state=TaskState.COMPLETED,
            created_at=now - timedelta(seconds=10),
            started_at=now - timedelta(seconds=5),
            completed_at=now,
        )

        # Duration should be ~5000ms
        assert task.duration_ms is not None
        assert 4900 <= task.duration_ms <= 5100


class TestBackgroundTaskManager:
    """Tests for BackgroundTaskManager."""

    def test_submit_simple_task(self, task_manager):
        """Test submitting a simple task."""

        def simple_task():
            return 42

        task_id = task_manager.submit(simple_task, name="Simple Task")

        assert task_id is not None
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.name == "Simple Task"

    def test_submit_task_with_args(self, task_manager):
        """Test submitting a task with arguments."""

        def add(a, b):
            return a + b

        task_id = task_manager.submit(add, args=(5, 3), name="Add Task")
        result = task_manager.wait_for_task(task_id, timeout=5.0)

        assert result is not None
        assert result.success
        assert result.value == 8

    def test_submit_task_with_kwargs(self, task_manager):
        """Test submitting a task with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        task_id = task_manager.submit(
            greet, args=("World",), kwargs={"greeting": "Hi"}, name="Greet Task"
        )
        result = task_manager.wait_for_task(task_id, timeout=5.0)

        assert result is not None
        assert result.success
        assert result.value == "Hi, World!"

    def test_task_failure_handling(self, task_manager):
        """Test handling of task failures."""

        def failing_task():
            raise ValueError("Intentional error")

        task_id = task_manager.submit(failing_task, name="Failing Task")
        result = task_manager.wait_for_task(task_id, timeout=5.0)

        assert result is not None
        assert not result.success
        assert "Intentional error" in result.error
        assert result.error_type == "ValueError"

        task = task_manager.get_task(task_id)
        assert task.state == TaskState.FAILED

    def test_task_with_tags_and_metadata(self, task_manager):
        """Test task with tags and metadata."""
        task_id = task_manager.submit(
            lambda: 1,
            name="Tagged Task",
            tags=["test", "example"],
            metadata={"key": "value"},
        )

        task = task_manager.get_task(task_id)
        assert "test" in task.tags
        assert "example" in task.tags
        assert task.metadata.get("key") == "value"

    def test_get_tasks_by_tag(self, task_manager):
        """Test retrieving tasks by tag."""
        task_manager.submit(lambda: 1, tags=["alpha"])
        task_manager.submit(lambda: 2, tags=["alpha", "beta"])
        task_manager.submit(lambda: 3, tags=["beta"])

        alpha_tasks = task_manager.get_tasks_by_tag("alpha")
        beta_tasks = task_manager.get_tasks_by_tag("beta")

        assert len(alpha_tasks) == 2
        assert len(beta_tasks) == 2

    def test_progress_update(self, task_manager):
        """Test progress updates."""
        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.to_dict())

        def task_with_progress(progress_reporter):
            for i in range(5):
                progress_reporter.update(current=i, total=5, message=f"Step {i}")
                time.sleep(0.01)
            return "done"

        task_id = task_manager.submit(
            task_with_progress,
            name="Progress Task",
            on_progress=on_progress,
        )
        task_manager.wait_for_task(task_id, timeout=5.0)

        assert len(progress_updates) >= 1
        assert progress_updates[-1]["current"] == 4  # Last update was i=4

    def test_completion_callback(self, task_manager):
        """Test completion callback."""
        callback_result = []

        def on_complete(result):
            callback_result.append(result)

        task_id = task_manager.submit(
            lambda: "success",
            name="Callback Task",
            on_complete=on_complete,
        )
        task_manager.wait_for_task(task_id, timeout=5.0)

        assert len(callback_result) == 1
        assert callback_result[0].success
        assert callback_result[0].value == "success"

    def test_cancel_task(self, task_manager):
        """Test task cancellation."""
        cancelled = []

        def long_task(progress_reporter):
            for i in range(100):
                if progress_reporter.check_cancelled():
                    cancelled.append(True)
                    return "cancelled"
                time.sleep(0.05)
            return "done"

        task_id = task_manager.submit(long_task, name="Long Task")

        # Give it a moment to start
        time.sleep(0.1)

        # Cancel the task
        result = task_manager.cancel(task_id)
        assert result is True

        # Wait for it to finish
        task_manager.wait_for_task(task_id, timeout=5.0)

        # Should have been cancelled
        task = task_manager.get_task(task_id)
        assert task._cancel_requested

    def test_get_active_tasks(self, task_manager):
        """Test getting active tasks."""

        def slow_task():
            time.sleep(1)
            return "done"

        task_id = task_manager.submit(slow_task, name="Slow Task")

        # Give it time to start
        time.sleep(0.1)

        active = task_manager.get_active_tasks()
        assert len(active) >= 1
        assert any(t.task_id == task_id for t in active)

        # Cancel and cleanup
        task_manager.cancel(task_id)

    def test_get_task_status_dict(self, task_manager):
        """Test getting task status as dictionary."""
        task_id = task_manager.submit(lambda: 42, name="Status Task")
        task_manager.wait_for_task(task_id, timeout=5.0)

        status = task_manager.get_task_status(task_id)

        assert status is not None
        assert status["task_id"] == task_id
        assert status["name"] == "Status Task"
        assert status["state"] == "completed"
        assert status["result"]["success"] is True
        assert status["result"]["value"] == 42

    def test_task_not_found(self, task_manager):
        """Test handling of non-existent task."""
        task = task_manager.get_task("nonexistent-id")
        assert task is None

        status = task_manager.get_task_status("nonexistent-id")
        assert status is None

    def test_cleanup_old_tasks(self, task_manager):
        """Test cleanup of old tasks."""
        from datetime import datetime, timedelta

        # Create a completed task
        task_id = task_manager.submit(lambda: 1)
        task_manager.wait_for_task(task_id, timeout=5.0)

        # Manually age the task
        task = task_manager.get_task(task_id)
        task.created_at = datetime.utcnow() - timedelta(hours=2)

        # Cleanup (max_age=1 hour)
        removed = task_manager.cleanup_old_tasks(max_age_hours=1)

        assert removed == 1
        assert task_manager.get_task(task_id) is None


class TestTaskContext:
    """Tests for task_context context manager."""

    def test_successful_context(self, task_manager):
        """Test successful task context."""
        with task_manager.task_context("Context Task") as reporter:
            reporter.update(current=50, total=100, message="Halfway")

        # Find the task
        tasks = [t for t in task_manager._tasks.values() if t.name == "Context Task"]
        assert len(tasks) == 1
        assert tasks[0].state == TaskState.COMPLETED

    def test_failed_context(self, task_manager):
        """Test failed task context."""
        with pytest.raises(ValueError):
            with task_manager.task_context("Failing Context") as reporter:
                raise ValueError("Context error")

        # Find the task
        tasks = [t for t in task_manager._tasks.values() if t.name == "Failing Context"]
        assert len(tasks) == 1
        assert tasks[0].state == TaskState.FAILED
        assert "Context error" in tasks[0].result.error


class TestProgressReporter:
    """Tests for ProgressReporter."""

    def test_progress_reporter_updates(self, task_manager):
        """Test progress reporter updates task."""
        task_id = task_manager.submit(lambda: 1)

        reporter = task_manager.progress_reporter(task_id)
        reporter.update(current=10, total=20, message="Test", phase="Phase A")

        task = task_manager.get_task(task_id)
        assert task.progress.current == 10
        assert task.progress.total == 20
        assert task.progress.message == "Test"
        assert task.progress.phase == "Phase A"

    def test_progress_reporter_context(self, task_manager):
        """Test progress reporter as context manager."""
        task_id = task_manager.submit(lambda: 1)

        with task_manager.progress_reporter(task_id) as reporter:
            reporter.update(current=5, total=10)
            assert not reporter.check_cancelled()

        task = task_manager.get_task(task_id)
        assert task.progress.current == 5
