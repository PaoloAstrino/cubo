"""
CUBO Thread Manager
Manages thread pools for async operations with proper lifecycle management.
"""

import time
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Callable, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class ThreadManager:
    """Manages thread pools with lifecycle tracking and cleanup."""

    def __init__(self, max_workers: int = 4, thread_name_prefix: str = "cubo"):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.active_futures = set()  # Use set to prevent duplicates
        self._shutdown = False

        logger.info(f"ThreadManager initialized with {max_workers} max workers and "
                    f"prefix '{thread_name_prefix}'")

    def submit_task(
        self,
        fn: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Future:
        """
        Submit a task to the thread pool with tracking.

        Args:
            fn: Function to execute
            *args: Positional arguments for the function
            timeout: Optional timeout in seconds
            **kwargs: Keyword arguments for the function

        Returns:
            Future object for the task
        """
        if self._shutdown:
            raise RuntimeError("ThreadManager is shutting down")

        task_wrapper = self._create_task_wrapper(fn, args, kwargs, timeout)
        future = self.executor.submit(task_wrapper)
        self._track_future(future)

        return future

    def _create_task_wrapper(
        self, fn: Callable, args: tuple, kwargs: dict,
        timeout: Optional[float]
    ) -> Callable:
        """Create a wrapper function that handles timeouts and exceptions."""
        def task_wrapper():
            try:
                if timeout:
                    return self._run_with_timeout(fn, args, kwargs, timeout)
                else:
                    return fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                raise

        return task_wrapper

    def _run_with_timeout(self, fn: Callable, args: tuple, kwargs: dict, timeout: float):
        """Run a function with timeout handling."""
        import threading

        result = [None]
        exception = [None]

        def run_with_timeout():
            try:
                result[0] = fn(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_with_timeout, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Task timed out after {timeout} seconds")
        if exception[0]:
            raise exception[0]
        return result[0]

    def _track_future(self, future: Future) -> None:
        """Track a future and set up cleanup when done."""
        self.active_futures.add(future)

        def cleanup_future(f):
            self.active_futures.discard(f)
            if f.exception():
                logger.error(f"Task completed with exception: {f.exception()}")

        future.add_done_callback(cleanup_future)

    def submit_task_with_retry(
        self,
        fn: Callable,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Future:
        """
        Submit a task with automatic retry on failure.

        Args:
            fn: Function to execute
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Optional timeout per attempt
        """

        def retry_wrapper():
            last_exception = None
            current_retry_delay = retry_delay  # Local copy for modification
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Task attempt {attempt + 1} failed: {e}. "
                                       f"Retrying in {current_retry_delay}s...")
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 1.5  # Exponential backoff
                    else:
                        logger.error(f"Task failed after {max_retries + 1} attempts: {e}")
                        raise last_exception

        return self.submit_task(retry_wrapper, timeout=timeout)

    def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all active tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        while self._should_continue_waiting(start_time, timeout):
            self._process_pending_futures()

        return self._all_tasks_completed()

    def _should_continue_waiting(self, start_time: float, timeout: Optional[float]) -> bool:
        """Check if we should continue waiting for tasks."""
        if not self.active_futures:
            return False

        if timeout is None:
            return True

        return time.time() - start_time < timeout

    def _process_pending_futures(self):
        """Process any futures that may have completed."""
        for future in list(self.active_futures):
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Future completed with error: {e}")

    def _all_tasks_completed(self) -> bool:
        """Check if all tasks have completed."""
        return len(self.active_futures) == 0

    def get_active_count(self) -> int:
        """Get the number of currently active tasks."""
        # Clean up completed futures
        self.active_futures = {f for f in self.active_futures if not f.done()}
        return len(self.active_futures)

    def get_status(self) -> dict:
        """Get status information about the thread manager."""
        return {
            "active_tasks": self.get_active_count(),
            "max_workers": self.executor._max_workers,
            "shutdown": self._shutdown
        }

    def shutdown(self, wait: bool = True):
        """Shutdown the thread manager."""
        logger.info("Shutting down ThreadManager...")
        self._shutdown = True

        if wait:
            self.wait_for_all(timeout=30.0)  # Wait up to 30 seconds

        self.executor.shutdown(wait=wait)
        logger.info("ThreadManager shutdown complete")

    @contextmanager
    def task_context(self, timeout: Optional[float] = None):
        """
        Context manager for executing tasks with automatic cleanup.

        Usage:
            with thread_manager.task_context(timeout=30):
                future = thread_manager.submit_task(my_function)
                result = future.result()
        """
        try:
            yield self
        finally:
            # Ensure cleanup happens even if context is exited early
            if not self._shutdown:
                self.wait_for_all(timeout=timeout or 10.0)
