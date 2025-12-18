"""
BackgroundTaskManager - Handles async task execution for non-blocking UI.

This manager wraps ThreadPoolExecutor to allow fire-and-forget tasks
(like long ingestion) while providing a mechanism to poll for status.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, Dict, Optional

from cubo.utils.logger import logger


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTaskManager:
    def __init__(self, max_workers: int = 2):
        # Keep workers low to avoid starving the application
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cubo_bg")
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a function to run in the background. Returns a Job ID."""
        job_id = str(uuid.uuid4())

        with self._lock:
            self._tasks[job_id] = {
                "status": TaskStatus.PENDING.value,
                "result": None,
                "error": None,
                "progress": 0.0,
            }

        self.executor.submit(self._run_task, job_id, func, *args, **kwargs)
        return job_id

    def _run_task(self, job_id: str, func: Callable, *args, **kwargs):
        """Internal wrapper to handle status updates."""
        with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id]["status"] = TaskStatus.RUNNING.value

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if job_id in self._tasks:
                    self._tasks[job_id]["status"] = TaskStatus.COMPLETED.value
                    self._tasks[job_id]["result"] = result
                    self._tasks[job_id]["progress"] = 1.0
        except Exception as e:
            logger.error(f"Background task {job_id} failed: {e}")
            with self._lock:
                if job_id in self._tasks:
                    self._tasks[job_id]["status"] = TaskStatus.FAILED.value
                    self._tasks[job_id]["error"] = str(e)

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job."""
        with self._lock:
            return self._tasks.get(job_id)

    def update_progress(self, job_id: str, progress: float) -> None:
        """Update progress (0.0 to 1.0) for a running task."""
        with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id]["progress"] = progress

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


# Global singleton
bg_manager = BackgroundTaskManager(max_workers=2)
