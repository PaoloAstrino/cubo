from src.cubo.workers.enhanced_thread_manager import *

__all__ = [name for name in dir() if not name.startswith('_')]
    """Advanced thread manager for CPU/GPU intensive tasks with proper resource management."""

    def __init__(self, max_cpu_workers: Optional[int] = None, max_io_workers: int = 4):
        """
        Initialize enhanced thread manager.

        Args:
            max_cpu_workers: Maximum CPU workers (default: CPU count, max 8)
            max_io_workers: Maximum I/O workers (default: 4)
        """
        self.cpu_count = psutil.cpu_count()
        self.max_cpu_workers = min(max_cpu_workers or self.cpu_count, 8)  # Cap at 8 for stability
        self.max_io_workers = max_io_workers

        # Thread pools for different task types
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_cpu_workers,
            thread_name_prefix="cpu_worker"
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_io_workers,
            thread_name_prefix="io_worker"
        )

        # GPU task queue (single thread for GPU operations to avoid context switching)
        self.gpu_queue = queue.Queue()
        self.gpu_thread = threading.Thread(target=self._gpu_worker, daemon=True)
        self.gpu_thread.start()

        # Task tracking
        self.active_tasks: Dict[int, Dict[str, Any]] = {}
        self.task_lock = threading.Lock()

        # Performance monitoring
        self.task_stats = {
            'cpu_tasks_completed': 0,
            'io_tasks_completed': 0,
            'gpu_tasks_completed': 0,
            'cpu_task_times': [],
            'io_task_times': [],
            'gpu_task_times': []
        }

        logger.info(f"EnhancedThreadManager initialized: CPU={self.max_cpu_workers}, I/O={self.max_io_workers}")

    def submit_cpu_task(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit CPU-intensive task (model inference, embeddings, computation)."""
        future = self.cpu_executor.submit(fn, *args, **kwargs)
        self._track_task(future, "cpu")
        return future

    def submit_io_task(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit I/O bound task (file reading, network operations, database)."""
        future = self.io_executor.submit(fn, *args, **kwargs)
        self._track_task(future, "io")
        return future

    def submit_gpu_task(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit GPU-intensive task (single-threaded to avoid context switching)."""
        future = concurrent.futures.Future()

        def gpu_wrapper():
            start_time = time.time()
            try:
                result = fn(*args, **kwargs)
                execution_time = time.time() - start_time
                future.set_result(result)
                self._record_task_completion("gpu", execution_time)
            except Exception as e:
                future.set_exception(e)
                logger.error(f"GPU task error: {e}")

        self.gpu_queue.put(gpu_wrapper)
        self._track_task(future, "gpu")
        return future

    def _gpu_worker(self):
        """Dedicated GPU worker thread."""
        logger.info("GPU worker thread started")
        while True:
            task = self.gpu_queue.get()
            if task is None:  # Poison pill for shutdown
                logger.info("GPU worker thread shutting down")
                break
            try:
                task()
            except Exception as e:
                logger.error(f"GPU worker error: {e}")
            finally:
                self.gpu_queue.task_done()

    def _track_task(self, future: concurrent.futures.Future, task_type: str):
        """Track active tasks for monitoring."""
        with self.task_lock:
            task_id = id(future)
            self.active_tasks[task_id] = {
                'future': future,
                'type': task_type,
                'start_time': time.time()
            }

            # Cleanup completed tasks periodically
            if len(self.active_tasks) > 100:  # Prevent memory buildup
                self._cleanup_completed_tasks()

    def _cleanup_completed_tasks(self):
        """Remove completed tasks from tracking."""
        with self.task_lock:
            completed = [tid for tid, info in self.active_tasks.items()
                         if info['future'].done()]
            for tid in completed:
                del self.active_tasks[tid]

    def _record_task_completion(self, task_type: str, execution_time: float):
        """Record task completion statistics."""
        with self.task_lock:
            self.task_stats[f'{task_type}_tasks_completed'] += 1
            self.task_stats[f'{task_type}_task_times'].append(execution_time)

            # Keep only last 100 times for memory efficiency
            times_list = self.task_stats[f'{task_type}_task_times']
            if len(times_list) > 100:
                times_list.pop(0)

    def get_active_task_count(self) -> Dict[str, int]:
        """Get count of active tasks by type."""
        with self.task_lock:
            counts = {'cpu': 0, 'io': 0, 'gpu': 0}
            for info in self.active_tasks.values():
                counts[info['type']] += 1
            return counts

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.task_lock:
            stats = self.task_stats.copy()

            # Calculate averages
            for task_type in ['cpu', 'io', 'gpu']:
                times = stats[f'{task_type}_task_times']
                if times:
                    stats[f'{task_type}_avg_time'] = sum(times) / len(times)
                else:
                    stats[f'{task_type}_avg_time'] = 0.0

            # Add current active tasks
            stats['active_tasks'] = self.get_active_task_count()

            return stats

    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """Wait for all active tasks to complete."""
        start_time = time.time()
        while self.active_tasks:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)  # Small delay to avoid busy waiting
        return True

    def shutdown(self, wait: bool = True):
        """Graceful shutdown of all thread pools."""
        logger.info("Shutting down EnhancedThreadManager...")

        # Stop GPU worker
        self.gpu_queue.put(None)
        self.gpu_thread.join(timeout=5.0)

        # Shutdown executors
        self.cpu_executor.shutdown(wait=wait)
        self.io_executor.shutdown(wait=wait)

        logger.info("EnhancedThreadManager shutdown complete")


# Global instance
_enhanced_thread_manager = None


def get_enhanced_thread_manager() -> EnhancedThreadManager:
    """Get the global enhanced thread manager instance."""
    global _enhanced_thread_manager
    if _enhanced_thread_manager is None:
        _enhanced_thread_manager = EnhancedThreadManager()
    return _enhanced_thread_manager
