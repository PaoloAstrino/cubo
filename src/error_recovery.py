"""
CUBO Error Recovery System
Provides error handling and recovery strategies for backend operations.
"""

import time
import logging
from typing import Callable, Any, Optional, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ESCALATE = "escalate"


class ErrorRecoveryManager:
    """
    Manages error recovery for different types of operations.
    Provides retry logic, fallback strategies, and error classification.
    """

    def __init__(self):
        self.recovery_configs = {
            'document_processing': {
                'strategy': RecoveryStrategy.RETRY,
                'max_retries': 3,
                'retry_delay': 1.0,
                'timeout': 300.0  # 5 minutes
            },
            'embedding_generation': {
                'strategy': RecoveryStrategy.RETRY,
                'max_retries': 2,
                'retry_delay': 2.0,
                'timeout': 120.0  # 2 minutes
            },
            'database_operation': {
                'strategy': RecoveryStrategy.RETRY,
                'max_retries': 5,
                'retry_delay': 1.0,
                'timeout': 60.0  # 1 minute
            },
            'llm_generation': {
                'strategy': RecoveryStrategy.FALLBACK,
                'max_retries': 2,
                'retry_delay': 3.0,
                'timeout': 180.0,  # 3 minutes
                'fallback_response': "I apologize, but I'm unable to generate a response at this time. Please try again."
            }
        }

        self.error_counts = {}
        self.last_errors = {}

    def execute_with_recovery(
        self,
        operation_type: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with error recovery.

        Args:
            operation_type: Type of operation (key in recovery_configs)
            operation: Callable to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If recovery fails
        """
        config = self.recovery_configs.get(operation_type, {
            'strategy': RecoveryStrategy.RETRY,
            'max_retries': 1,
            'retry_delay': 1.0,
            'timeout': 30.0
        })

        last_exception = None

        for attempt in range(config['max_retries'] + 1):
            try:
                # Track error counts
                self._record_attempt(operation_type, attempt == 0)

                # Execute with timeout
                result = self._execute_with_timeout(
                    operation, config['timeout'], *args, **kwargs
                )

                # Success - reset error counts
                self._record_success(operation_type)
                return result

            except Exception as e:
                last_exception = e
                self._record_error(operation_type, e)

                if attempt < config['max_retries']:
                    logger.warning(
                        f"{operation_type} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {config['retry_delay']}s..."
                    )
                    time.sleep(config['retry_delay'])
                    config['retry_delay'] *= 1.5  # Exponential backoff
                else:
                    logger.error(
                        f"{operation_type} failed after {config['max_retries'] + 1} attempts: {e}"
                    )

                    # Try fallback strategy
                    if config['strategy'] == RecoveryStrategy.FALLBACK:
                        return self._execute_fallback(config, e)
                    elif config['strategy'] == RecoveryStrategy.SKIP:
                        logger.info(f"Skipping {operation_type} due to error: {e}")
                        return None

                    # Re-raise the last exception
                    raise last_exception

    def _execute_with_timeout(self, operation: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute operation with timeout."""
        import threading

        result = [None]
        exception = [None]
        completed = [False]

        def run_operation():
            try:
                result[0] = operation(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                exception[0] = e
                completed[0] = True

        thread = threading.Thread(target=run_operation, daemon=True)
        thread.start()
        thread.join(timeout)

        if not completed[0]:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        if exception[0]:
            raise exception[0]

        return result[0]

    def _execute_fallback(self, config: Dict, original_error: Exception) -> Any:
        """Execute fallback strategy."""
        logger.info(f"Executing fallback for error: {original_error}")
        return config.get('fallback_response', None)

    def _record_attempt(self, operation_type: str, is_first_attempt: bool):
        """Record an attempt for monitoring."""
        if operation_type not in self.error_counts:
            self.error_counts[operation_type] = {
                'total_attempts': 0,
                'failures': 0,
                'successes': 0,
                'last_failure': None
            }

        self.error_counts[operation_type]['total_attempts'] += 1

    def _record_success(self, operation_type: str):
        """Record a successful operation."""
        if operation_type in self.error_counts:
            self.error_counts[operation_type]['successes'] += 1

    def _record_error(self, operation_type: str, error: Exception):
        """Record an error for monitoring."""
        if operation_type not in self.error_counts:
            self.error_counts[operation_type] = {
                'total_attempts': 0,
                'failures': 0,
                'successes': 0,
                'last_failure': None
            }

        self.error_counts[operation_type]['failures'] += 1
        self.error_counts[operation_type]['last_failure'] = time.time()
        self.last_errors[operation_type] = str(error)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all operations."""
        status = {}

        for operation_type, counts in self.error_counts.items():
            failure_rate = counts['failures'] / max(counts['total_attempts'], 1)
            recent_failure = False

            if counts['last_failure']:
                # Consider recent if within last 5 minutes
                recent_failure = (time.time() - counts['last_failure']) < 300

            status[operation_type] = {
                'healthy': failure_rate < 0.5 and not recent_failure,
                'failure_rate': failure_rate,
                'total_attempts': counts['total_attempts'],
                'recent_failure': recent_failure,
                'last_error': self.last_errors.get(operation_type)
            }

        return status

    def reset_error_counts(self, operation_type: Optional[str] = None):
        """Reset error counts for monitoring."""
        if operation_type:
            self.error_counts.pop(operation_type, None)
            self.last_errors.pop(operation_type, None)
        else:
            self.error_counts.clear()
            self.last_errors.clear()

    def add_recovery_config(self, operation_type: str, config: Dict):
        """Add or update recovery configuration for an operation type."""
        self.recovery_configs[operation_type] = config
        logger.info(f"Updated recovery config for {operation_type}: {config}")
