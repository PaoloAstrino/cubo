"""
CUBO Service Manager
Orchestrates thread management, error recovery, and health monitoring
for backend operations.
"""

import time
import logging
import sqlite3
from typing import Any, Dict, Optional, Callable, List
from contextlib import contextmanager

from src.cubo.workers.thread_manager import ThreadManager
from src.cubo.utils.error_recovery import ErrorRecoveryManager
from src.cubo.monitoring.health_monitor import HealthMonitor, HealthStatus

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Central service manager that coordinates thread pools, error recovery,
    and health monitoring. Provides a unified interface for backend
    operations with reliability features.
    """

    def __init__(self, max_workers: int = 4):
        self.thread_manager = ThreadManager(max_workers=max_workers)
        self.error_recovery = ErrorRecoveryManager()
        self.health_monitor = HealthMonitor()

        # Register health checks for backend components
        self._register_component_health_checks()

        # Set up health alert handling
        self.health_monitor.add_alert_callback(self._handle_health_alert)

        logger.info("ServiceManager initialized")

    def execute_async(
        self,
        operation_type: str,
        operation: Callable,
        *args,
        with_retry: bool = True,
        **kwargs
    ):
        """
        Execute an operation asynchronously with error recovery.

        Args:
            operation_type: Type of operation for recovery configuration
            operation: Function to execute
            with_retry: Whether to use retry logic
            *args, **kwargs: Arguments for the operation

        Returns:
            Future object
        """

        def wrapped_operation():
            return self.error_recovery.execute_with_recovery(
                operation_type, operation, *args, **kwargs
            )

        if with_retry:
            return self.thread_manager.submit_task_with_retry(
                wrapped_operation,
                max_retries=self.error_recovery.recovery_configs
                .get(operation_type, {}).get('max_retries', 1)
            )
        else:
            return self.thread_manager.submit_task(wrapped_operation)

    def execute_sync(
        self,
        operation_type: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation synchronously with error recovery.

        Args:
            operation_type: Type of operation for recovery configuration
            operation: Function to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation
        """
        return self.error_recovery.execute_with_recovery(
            operation_type, operation, *args, **kwargs
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'threads': self.thread_manager.get_status(),
            'health': self.health_monitor.get_health_status(),
            'errors': self.error_recovery.get_health_status(),
            'timestamp': time.time()
        }

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all active operations to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all operations completed within timeout
        """
        return self.thread_manager.wait_for_all(timeout)

    def shutdown(self, wait: bool = True):
        """Shutdown all services gracefully."""
        logger.info("Shutting down ServiceManager...")

        # Wait for operations to complete
        if wait:
            self.wait_for_completion(timeout=30.0)

        # Shutdown components
        self.thread_manager.shutdown(wait=wait)

        logger.info("ServiceManager shutdown complete")

    @contextmanager
    def operation_context(self, timeout: Optional[float] = None):
        """
        Context manager for executing operations with automatic cleanup.

        Usage:
            with service_manager.operation_context():
                future = service_manager.execute_async(
                    'document_processing', my_function
                )
                result = future.result()
        """
        try:
            yield self
        finally:
            # Ensure cleanup happens even if context is exited early
            if hasattr(self.thread_manager, '_shutdown') and \
                    not self.thread_manager._shutdown:
                self.thread_manager.wait_for_all(timeout=timeout or 10.0)

    def _register_component_health_checks(self):
        """Register health checks for backend components."""
        # Thread pool health
        self.health_monitor.add_health_check(
            "thread_pool",
            self._check_thread_pool_health,
            interval=30.0
        )

        # Error recovery health
        self.health_monitor.add_health_check(
            "error_recovery",
            self._check_error_recovery_health,
            interval=60.0
        )

    def _check_thread_pool_health(self) -> Dict[str, Any]:
        """Check thread pool health."""
        status = self.thread_manager.get_status()
        active_tasks = status['active_tasks']
        max_workers = status['max_workers']

        if active_tasks > max_workers * 0.9:  # Over 90% capacity
            health_status = HealthStatus.WARNING
            message = (
                "Thread pool near capacity: "
                f"{active_tasks}/{max_workers} active"
            )
        elif active_tasks > max_workers * 0.5:  # Over 50% capacity
            health_status = HealthStatus.HEALTHY
            message = (
                "Thread pool active: "
                f"{active_tasks}/{max_workers} active"
            )
        else:
            health_status = HealthStatus.HEALTHY
            message = (
                "Thread pool healthy: "
                f"{active_tasks}/{max_workers} active"
            )

        return {
            'status': health_status.value,
            'message': message,
            'active_tasks': active_tasks,
            'max_workers': max_workers,
            'utilization': (
                active_tasks / max_workers if max_workers > 0 else 0
            ),
            'timestamp': time.time()
        }

    def _check_error_recovery_health(self) -> Dict[str, Any]:
        """Check error recovery system health."""
        error_status = self.error_recovery.get_health_status()

        # Check if any operations have high failure rates
        critical_operations = []
        warning_operations = []

        for op_type, status in error_status.items():
            if not status['healthy']:
                if status['failure_rate'] > 0.5 or status['recent_failure']:
                    critical_operations.append(op_type)
                else:
                    warning_operations.append(op_type)

        if critical_operations:
            health_status = HealthStatus.CRITICAL
            message = (
                "Critical error rates in: "
                f"{', '.join(critical_operations)}"
            )
        elif warning_operations:
            health_status = HealthStatus.WARNING
            message = f"High error rates in: {', '.join(warning_operations)}"
        else:
            health_status = HealthStatus.HEALTHY
            message = "Error recovery system healthy"

        return {
            'status': health_status.value,
            'message': message,
            'critical_operations': critical_operations,
            'warning_operations': warning_operations,
            'operation_status': error_status,
            'timestamp': time.time()
        }

    def _handle_health_alert(
        self, check_name: str, status: HealthStatus, details: Dict
    ):
        """Handle health alerts."""
        if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            logger.warning(f"Health alert - {check_name}: {status.value} - "
                           f"{details.get('message', '')}")
        elif status == HealthStatus.HEALTHY:
            logger.info(
                f"Health recovered - {check_name}: {details.get('message', '')}"
            )

    # Convenience methods for common operations

    def process_document_async(
        self, filepath: str, processor_func: Callable, *args, **kwargs
    ):
        """Process a document asynchronously with error recovery."""
        return self.execute_async(
            'document_processing', processor_func, filepath, *args, **kwargs
        )

    def generate_embedding_async(
        self, text: str, generator_func: Callable, *args, **kwargs
    ):
        """Generate embeddings asynchronously with error recovery."""
        return self.execute_async(
            'embedding_generation', generator_func, text, *args, **kwargs
        )

    def query_database_async(self, query_func: Callable, *args, **kwargs):
        """Query database asynchronously with error recovery."""
        return self.execute_async(
            'database_operation', query_func, *args, **kwargs
        )

    def generate_response_async(
        self, query: str, context: str, generator_func: Callable,
        sources: List[str], *args, **kwargs
    ):
        """Generate LLM response asynchronously with error recovery and automatic data saving."""
        import time
        start_time = time.time()

        # First execute the generation
        future = self.execute_async('llm_generation', generator_func, query, context,
                                    *args, **kwargs)

        # Add data saving callback after generation completes

        def on_generation_complete(f):
            if not f.exception():
                try:
                    # Extract result from future (which is just the response string)
                    response = f.result()
                    # Calculate actual response time
                    response_time = time.time() - start_time
                    # Save query data with provided sources and calculated response_time
                    self._save_query_data(query, response, sources, response_time)
                except Exception as e:
                    logger.error(f"Failed to save query data after generation: {e}")

        future.add_done_callback(on_generation_complete)
        return future

    def _save_query_data(self, question: str, answer: str, sources: List[str],
                         response_time: float):
        """Save query data without evaluation in background thread."""
        try:

            def save_data():
                try:
                    # Direct database saving without evaluation
                    success = self._save_query_data_direct(
                        question=question,
                        answer=answer,
                        contexts=sources,
                        response_time=response_time
                    )

                    if success:
                        logger.info(f"Query data saved successfully: {question[:50]}...")
                    else:
                        logger.error(f"Failed to save query data: {question[:50]}...")

                    return success

                except Exception as e:
                    logger.error(f"Background data saving failed: {e}")
                    return False

            # Execute data saving in background (non-blocking)
            self.execute_async('data_saving', save_data, with_retry=False)

        except Exception as e:
            logger.error(f"Failed to schedule data saving: {e}")

    def _save_query_data_direct(
        self, question: str, answer: str, contexts: List[str],
        response_time: float
    ) -> bool:
        """Save query data directly to database without evaluation."""
        try:
            # Create evaluation data
            evaluation_data = self._create_evaluation_data(
                question, answer, contexts, response_time
            )

            # Ensure database setup
            db_path = "evaluation/evaluation.db"
            self._ensure_database_directory(db_path)

            # Save to database
            self._save_to_database(db_path, evaluation_data)

            return True

        except Exception as e:
            logger.error(f"Direct data saving failed: {e}")
            return False

    def _create_evaluation_data(
        self, question: str, answer: str, contexts: List[str],
        response_time: float, context_metadata: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create evaluation data dictionary."""
        import json
        import datetime
        import uuid

        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': str(uuid.uuid4())[:8],
            'question': question,
            'answer': answer,
            'response_time': response_time,
            'contexts': json.dumps(contexts),
            'context_metadata': json.dumps(context_metadata or []),
            'model_used': 'llama3.2:latest',
            'embedding_model': 'embeddinggemma-300m',
            'retrieval_method': 'sentence_window',
            'chunking_method': 'sentence_window',
            'answer_length': len(answer) if answer else 0,
            'context_count': len(contexts),
            'total_context_length': sum(len(ctx) for ctx in contexts),
            'average_context_similarity': 0.0,
            'answer_confidence': 0.0,
            'has_answer': bool(answer and not answer.startswith("Error")),
            'is_fallback_response': bool(answer and "unable to generate" in answer.lower()),
            'error_occurred': False,
            'error_message': None
        }

    def _ensure_database_directory(self, db_path: str) -> None:
        """Ensure the database directory exists."""
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def _save_to_database(self, db_path: str, evaluation_data: Dict[str, Any]) -> None:
        """Save evaluation data to database."""
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            # Create table if it doesn't exist
            self._create_evaluation_table(conn)

            # Add missing columns for backward compatibility
            self._add_missing_columns(conn)

            # Insert data
            self._insert_evaluation_data(conn, evaluation_data)

            conn.commit()

    def _create_evaluation_table(self, conn: sqlite3.Connection) -> None:
        """Create the evaluations table if it doesn't exist."""
        conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT,
                response_time REAL,
                contexts TEXT,  -- JSON array
                context_metadata TEXT,  -- JSON array
                model_used TEXT,
                embedding_model TEXT,
                retrieval_method TEXT,
                chunking_method TEXT,

                -- RAG Metrics
                answer_relevance_score REAL,
                context_relevance_score REAL,
                groundedness_score REAL,

                -- Detailed Metrics
                answer_length INTEGER,
                context_count INTEGER,
                total_context_length INTEGER,
                average_context_similarity REAL,
                answer_confidence REAL,

                -- Quality Flags
                has_answer BOOLEAN,
                is_fallback_response BOOLEAN,
                error_occurred BOOLEAN,
                error_message TEXT,

                -- LLM Metrics
                llm_metrics TEXT,

                -- User Feedback
                user_rating INTEGER,
                user_feedback TEXT
            )
        ''')

    def _add_missing_columns(self, conn: sqlite3.Connection) -> None:
        """Add missing columns for backward compatibility."""
        columns_to_add = [
            'answer_relevance_score REAL',
            'context_relevance_score REAL',
            'groundedness_score REAL',
            'llm_metrics TEXT',
            'user_rating INTEGER',
            'user_feedback TEXT'
        ]

        for column_def in columns_to_add:
            try:
                conn.execute(f'ALTER TABLE evaluations ADD COLUMN {column_def}')
            except sqlite3.OperationalError:
                pass  # Column already exists

    def _insert_evaluation_data(
        self, conn: sqlite3.Connection, evaluation_data: Dict[str, Any]
    ) -> None:
        """Insert evaluation data into the database."""
        conn.execute('''
            INSERT INTO evaluations (
                timestamp, session_id, question, answer, response_time,
                contexts, context_metadata, model_used, embedding_model,
                retrieval_method, chunking_method, answer_length,
                context_count, total_context_length, average_context_similarity,
                answer_confidence, has_answer, is_fallback_response,
                error_occurred, error_message,
                answer_relevance_score, context_relevance_score, groundedness_score,
                llm_metrics, user_rating, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation_data['timestamp'],
            evaluation_data['session_id'],
            evaluation_data['question'],
            evaluation_data['answer'],
            evaluation_data['response_time'],
            evaluation_data['contexts'],
            evaluation_data['context_metadata'],
            evaluation_data['model_used'],
            evaluation_data['embedding_model'],
            evaluation_data['retrieval_method'],
            evaluation_data['chunking_method'],
            evaluation_data['answer_length'],
            evaluation_data['context_count'],
            evaluation_data['total_context_length'],
            evaluation_data['average_context_similarity'],
            evaluation_data['answer_confidence'],
            evaluation_data['has_answer'],
            evaluation_data['is_fallback_response'],
            evaluation_data['error_occurred'],
            evaluation_data['error_message'],
            None,  # answer_relevance_score
            None,  # context_relevance_score
            None,  # groundedness_score
            None,  # llm_metrics
            None,  # user_rating
            None   # user_feedback
        ))


# Global service manager instance
_service_manager = None


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


def shutdown_service_manager():
    """Shutdown the global service manager."""
    global _service_manager
    if _service_manager:
        _service_manager.shutdown()
        _service_manager = None
