"""
Background Task Manager - Unified management of async/background operations.

This module provides a centralized system for managing background tasks with:
- Task submission and tracking
- Progress monitoring with callbacks
- Cancellation support
- Resource cleanup
- Integration with existing JobManager for API status tracking

The BackgroundTaskManager acts as a facade over the thread pool and job tracking,
providing a clean interface for long-running operations like document ingestion.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from src.cubo.config import config
from src.cubo.utils.logger import logger

T = TypeVar("T")


class TaskState(str, Enum):
    """State of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Priority levels for task scheduling."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskProgress:
    """Progress information for a task."""

    current: int = 0
    total: int = 100
    message: str = ""
    phase: str = ""

    @property
    def percentage(self) -> float:
        """Get progress as percentage."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "message": self.message,
            "phase": self.phase,
        }


@dataclass
class TaskResult(Generic[T]):
    """Result of a completed task."""

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "value": self.value,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class BackgroundTask:
    """Represents a background task with full lifecycle tracking."""

    task_id: str
    name: str
    state: TaskState
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    result: Optional[TaskResult] = None
    priority: TaskPriority = TaskPriority.NORMAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _future: Optional[Future] = field(default=None, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "state": self.state.value,
            "priority": self.priority.name,
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @property
    def is_complete(self) -> bool:
        """Check if task has finished (success, failure, or cancelled)."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT,
        )

    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.state == TaskState.RUNNING

    @property
    def duration_ms(self) -> Optional[float]:
        """Get task duration in milliseconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds() * 1000


class ProgressReporter:
    """
    Context manager for reporting task progress.

    Usage:
        with task_manager.progress_reporter(task_id) as reporter:
            for i, item in enumerate(items):
                reporter.update(current=i, total=len(items), message=f"Processing {item}")
                process(item)
    """

    def __init__(self, task_manager: BackgroundTaskManager, task_id: str):
        self._manager = task_manager
        self._task_id = task_id

    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Update task progress."""
        self._manager.update_progress(
            self._task_id,
            current=current,
            total=total,
            message=message,
            phase=phase,
        )

    def check_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._manager.is_cancelled(self._task_id)

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class BackgroundTaskManager:
    """
    Unified manager for background tasks with lifecycle tracking.

    Features:
    - Thread pool management with configurable workers
    - Task state tracking (pending, running, completed, failed, cancelled)
    - Progress reporting with callbacks
    - Cancellation support
    - Automatic cleanup of old tasks
    - Integration with JobManager for API status

    Usage:
        manager = BackgroundTaskManager()

        # Submit a task
        task_id = manager.submit(
            process_documents,
            args=(folder_path,),
            name="Document Ingestion",
            tags=["ingestion", "documents"],
        )

        # Check status
        task = manager.get_task(task_id)
        print(f"Progress: {task.progress.percentage}%")

        # Cancel if needed
        manager.cancel(task_id)
    """

    _instance: Optional[BackgroundTaskManager] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_task_age_hours: int = 24,
        thread_name_prefix: str = "cubo-bg",
    ):
        """
        Initialize the background task manager.

        Args:
            max_workers: Maximum concurrent workers (default from config)
            max_task_age_hours: Hours to keep completed tasks
            thread_name_prefix: Prefix for thread names
        """
        if self._initialized:
            return

        self._max_workers = max_workers or config.get("background_tasks.max_workers", 4)
        self._max_task_age_hours = max_task_age_hours
        self._thread_name_prefix = thread_name_prefix

        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=thread_name_prefix,
        )

        self._tasks: Dict[str, BackgroundTask] = {}
        self._tasks_lock = threading.Lock()
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        self._shutdown = False

        self._initialized = True
        logger.info(f"BackgroundTaskManager initialized with {self._max_workers} workers")

    def submit(
        self,
        fn: Callable[..., T],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        on_complete: Optional[Callable[[TaskResult], None]] = None,
    ) -> str:
        """
        Submit a task for background execution.

        Args:
            fn: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            name: Human-readable task name
            priority: Task priority level
            tags: Tags for categorization
            metadata: Additional metadata
            timeout: Timeout in seconds (None for no timeout)
            on_progress: Callback for progress updates
            on_complete: Callback when task completes

        Returns:
            task_id: Unique identifier for the task
        """
        if self._shutdown:
            raise RuntimeError("BackgroundTaskManager is shutting down")

        task_id = str(uuid.uuid4())
        kwargs = kwargs or {}

        task = BackgroundTask(
            task_id=task_id,
            name=name or fn.__name__,
            state=TaskState.PENDING,
            created_at=datetime.utcnow(),
            priority=priority,
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._tasks_lock:
            self._tasks[task_id] = task
            if on_progress:
                self._progress_callbacks[task_id] = [on_progress]

        # Create wrapper that handles lifecycle
        def task_wrapper():
            return self._execute_task(task_id, fn, args, kwargs, timeout, on_complete)

        future = self._executor.submit(task_wrapper)
        task._future = future

        logger.info(f"Submitted task {task_id}: {task.name}")
        return task_id

    def _execute_task(
        self,
        task_id: str,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        timeout: Optional[float],
        on_complete: Optional[Callable],
    ) -> Any:
        """Execute a task with full lifecycle management."""
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Mark as running
        task.state = TaskState.RUNNING
        task.started_at = datetime.utcnow()

        start_time = time.time()
        result_value = None
        error_msg = None
        error_type = None

        try:
            # Inject progress reporter if function accepts it
            if "progress_reporter" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                reporter = ProgressReporter(self, task_id)
                kwargs["progress_reporter"] = reporter

            if timeout:
                result_value = self._run_with_timeout(fn, args, kwargs, timeout, task_id)
            else:
                result_value = fn(*args, **kwargs)

            # Check if cancelled during execution
            if task._cancel_requested:
                task.state = TaskState.CANCELLED
            else:
                task.state = TaskState.COMPLETED

        except FutureTimeoutError:
            task.state = TaskState.TIMEOUT
            error_msg = f"Task timed out after {timeout} seconds"
            error_type = "TimeoutError"
            logger.warning(f"Task {task_id} timed out")

        except Exception as e:
            task.state = TaskState.FAILED
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Task {task_id} failed: {e}")

        finally:
            execution_time = (time.time() - start_time) * 1000
            task.completed_at = datetime.utcnow()

            task.result = TaskResult(
                success=task.state == TaskState.COMPLETED,
                value=result_value if task.state == TaskState.COMPLETED else None,
                error=error_msg,
                error_type=error_type,
                execution_time_ms=execution_time,
            )

            # Fire completion callback
            if on_complete:
                try:
                    on_complete(task.result)
                except Exception as e:
                    logger.error(f"Error in completion callback for {task_id}: {e}")

            logger.info(
                f"Task {task_id} finished with state {task.state.value} in {execution_time:.1f}ms"
            )

        return result_value

    def _run_with_timeout(
        self,
        fn: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float,
        task_id: str,
    ) -> Any:
        """Run function with timeout using a separate thread."""
        result = [None]
        exception = [None]

        def run_fn():
            try:
                result[0] = fn(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        # Copy context for proper trace_id propagation
        try:
            import contextvars

            ctx = contextvars.copy_context()
            thread = threading.Thread(target=lambda: ctx.run(run_fn), daemon=True)
        except Exception:
            thread = threading.Thread(target=run_fn, daemon=True)

        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Mark as cancelled to help the thread know to stop
            task = self._tasks.get(task_id)
            if task:
                task._cancel_requested = True
            raise FutureTimeoutError(f"Task timed out after {timeout} seconds")

        if exception[0]:
            raise exception[0]

        return result[0]

    def update_progress(
        self,
        task_id: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> bool:
        """
        Update task progress.

        Args:
            task_id: Task identifier
            current: Current progress value
            total: Total progress value
            message: Progress message
            phase: Current phase name

        Returns:
            True if updated, False if task not found
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if current is not None:
                task.progress.current = current
            if total is not None:
                task.progress.total = total
            if message is not None:
                task.progress.message = message
            if phase is not None:
                task.progress.phase = phase

            # Fire progress callbacks
            callbacks = self._progress_callbacks.get(task_id, [])
            for callback in callbacks:
                try:
                    callback(task.progress)
                except Exception as e:
                    logger.error(f"Error in progress callback for {task_id}: {e}")

            return True

    def progress_reporter(self, task_id: str) -> ProgressReporter:
        """
        Get a progress reporter for a task.

        Args:
            task_id: Task identifier

        Returns:
            ProgressReporter context manager
        """
        return ProgressReporter(self, task_id)

    def is_cancelled(self, task_id: str) -> bool:
        """Check if cancellation was requested for a task."""
        task = self._tasks.get(task_id)
        return task._cancel_requested if task else False

    def cancel(self, task_id: str, wait: bool = False, timeout: float = 5.0) -> bool:
        """
        Request cancellation of a task.

        Args:
            task_id: Task identifier
            wait: Whether to wait for task to stop
            timeout: How long to wait if wait=True

        Returns:
            True if cancellation requested, False if task not found
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        task._cancel_requested = True

        # Try to cancel the future
        if task._future and not task._future.done():
            task._future.cancel()

        if wait and task._future:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass

        logger.info(f"Cancellation requested for task {task_id}")
        return True

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status as dictionary for API response."""
        task = self.get_task(task_id)
        return task.to_dict() if task else None

    def get_active_tasks(self) -> List[BackgroundTask]:
        """Get all non-completed tasks."""
        with self._tasks_lock:
            return [t for t in self._tasks.values() if not t.is_complete]

    def get_tasks_by_tag(self, tag: str) -> List[BackgroundTask]:
        """Get all tasks with a specific tag."""
        with self._tasks_lock:
            return [t for t in self._tasks.values() if tag in t.tags]

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task identifier
            timeout: Maximum time to wait (None for infinite)

        Returns:
            TaskResult if completed, None if not found or timeout
        """
        task = self._tasks.get(task_id)
        if not task or not task._future:
            return None

        try:
            task._future.result(timeout=timeout)
            return task.result
        except Exception:
            return task.result

    def cleanup_old_tasks(self, max_age_hours: Optional[int] = None) -> int:
        """
        Remove completed tasks older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours (default from init)

        Returns:
            Number of tasks removed
        """
        from datetime import timedelta

        max_age = max_age_hours or self._max_task_age_hours
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age)

        with self._tasks_lock:
            tasks_to_remove = [
                task_id
                for task_id, task in self._tasks.items()
                if task.is_complete and task.created_at < cutoff_time
            ]

            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                self._progress_callbacks.pop(task_id, None)

            return len(tasks_to_remove)

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Shutdown the task manager.

        Args:
            wait: Whether to wait for running tasks
            timeout: Maximum time to wait for shutdown
        """
        self._shutdown = True

        # Cancel all pending tasks
        for task in self.get_active_tasks():
            self.cancel(task.task_id)

        # Shutdown executor
        self._executor.shutdown(wait=wait, cancel_futures=True)
        logger.info("BackgroundTaskManager shut down")

    @contextmanager
    def task_context(
        self,
        name: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for running code as a tracked task.

        Usage:
            with task_manager.task_context("My Operation") as ctx:
                ctx.update_progress(0, 100, "Starting")
                do_work()
                ctx.update_progress(100, 100, "Done")
        """
        task_id = str(uuid.uuid4())

        task = BackgroundTask(
            task_id=task_id,
            name=name,
            state=TaskState.RUNNING,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            tags=tags or [],
            metadata=metadata or {},
        )

        with self._tasks_lock:
            self._tasks[task_id] = task

        reporter = ProgressReporter(self, task_id)
        start_time = time.time()

        try:
            yield reporter
            task.state = TaskState.COMPLETED
            task.result = TaskResult(success=True)
        except Exception as e:
            task.state = TaskState.FAILED
            task.result = TaskResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            task.completed_at = datetime.utcnow()
            if task.result:
                task.result.execution_time_ms = (time.time() - start_time) * 1000


# Global instance accessor
_task_manager: Optional[BackgroundTaskManager] = None


def get_background_task_manager() -> BackgroundTaskManager:
    """Get the global background task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager
