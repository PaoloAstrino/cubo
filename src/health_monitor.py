"""
CUBO Health Monitor
Monitors system health and component status with alerting capabilities.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from src.exceptions import HealthCheckError, ServiceError, CUBOError
from src.logger import logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval: float = 30.0  # Check every 30 seconds
    timeout: float = 10.0   # Timeout after 10 seconds
    last_check: float = 0.0
    last_result: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """
    Monitors system and component health with configurable checks.
    Provides status reporting and alerting capabilities.
    """

    def __init__(self,
                 memory_warning_threshold: float = 80.0,
                 memory_critical_threshold: float = 90.0,
                 cpu_warning_threshold: float = 85.0,
                 cpu_critical_threshold: float = 95.0,
                 disk_warning_threshold: float = 90.0,
                 disk_critical_threshold: float = 95.0,
                 disk_check_path: str = '/'):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_callbacks: List[Callable[[str, HealthStatus, Dict], None]] = []
        self.status_history: Dict[str, List[Dict]] = {}

        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        self.cpu_warning_threshold = cpu_warning_threshold
        self.cpu_critical_threshold = cpu_critical_threshold
        self.disk_warning_threshold = disk_warning_threshold
        self.disk_critical_threshold = disk_critical_threshold
        self.disk_check_path = disk_check_path

        # Add default system health checks
        self.add_health_check("system_memory", self._check_system_memory, interval=60.0)
        self.add_health_check("system_cpu", self._check_system_cpu, interval=30.0)
        self.add_health_check("disk_space", self._check_disk_space, interval=300.0)  # 5 minutes

        logger.info("HealthMonitor initialized")

    def add_health_check(
        self,
        name: str,
        check_function: Callable[[], Dict[str, Any]],
        interval: float = 30.0,
        timeout: float = 10.0
    ):
        """Add a health check."""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout
        )
        self.status_history[name] = []
        logger.info(f"Added health check: {name}")

    def add_alert_callback(self, callback: Callable[[str, HealthStatus, Dict], None]):
        """Add a callback for health alerts."""
        self.alert_callbacks.append(callback)

    def perform_health_check(self, check_name: str) -> Dict[str, Any]:
        """Perform a specific health check."""
        # Validate check exists
        check = self._validate_check_exists(check_name)
        if not check:
            return self._create_unknown_check_result(check_name)

        try:
            # Execute check and process results
            result = self._execute_and_process_check(check_name, check)
            return result

        except Exception as e:
            # Handle check execution errors
            return self._handle_check_error(check_name, check, e)

    def _validate_check_exists(self, check_name: str) -> Optional[HealthCheck]:
        """Validate that a health check exists and return it."""
        return self.health_checks.get(check_name)

    def _create_unknown_check_result(self, check_name: str) -> Dict[str, Any]:
        """Create result for unknown health check."""
        return {
            'status': HealthStatus.UNKNOWN.value,
            'message': f'Health check {check_name} not found',
            'timestamp': time.time()
        }

    def _execute_and_process_check(self, check_name: str, check: HealthCheck) -> Dict[str, Any]:
        """Execute health check and process the results."""
        # Execute check with timeout
        result = self._execute_check_with_timeout(check)

        # Update check metadata and history
        self._update_check_metadata(check_name, check, result)

        # Trigger alerts if status changed
        self._check_for_alerts(check_name, result)

        return result

    def _update_check_metadata(self, check_name: str, check: HealthCheck, result: Dict[str, Any]):
        """Update check metadata and store in history."""
        check.last_check = time.time()
        check.last_result = result

        # Store in history (keep last 10 results)
        self.status_history[check_name].append(result)
        if len(self.status_history[check_name]) > 10:
            self.status_history[check_name].pop(0)

    def _handle_check_error(self, check_name: str, check: HealthCheck, error: Exception) -> Dict[str, Any]:
        """Handle errors during health check execution."""
        error_result = {
            'status': HealthStatus.CRITICAL.value,
            'message': f'Health check failed: {str(error)}',
            'timestamp': time.time(),
            'error': str(error)
        }

        check.last_result = error_result
        self._check_for_alerts(check_name, error_result)

        return error_result

    def get_health_status(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for all checks or a specific check."""
        if check_name:
            return self._get_single_health_status(check_name)
        else:
            return self._get_all_health_statuses()

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        for check_name in self.health_checks:
            results[check_name] = self.perform_health_check(check_name)

        return results

    def _execute_check_with_timeout(self, check: HealthCheck) -> Dict[str, Any]:
        """Execute a health check with timeout."""
        import threading

        result = [None]
        exception = [None]
        completed = [False]

        def run_check():
            try:
                result[0] = check.check_function()
                completed[0] = True
            except Exception as e:
                exception[0] = e
                completed[0] = True

        thread = threading.Thread(target=run_check, daemon=True)
        thread.start()
        thread.join(check.timeout)

        if not completed[0]:
            raise TimeoutError(f"Health check '{check.name}' timed out after "
                               f"{check.timeout} seconds")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _check_for_alerts(self, check_name: str, result: Dict[str, Any]):
        """Check if alerts should be triggered."""
        current_status = HealthStatus(result['status'])

        # Get previous status
        history = self.status_history[check_name]
        previous_status = HealthStatus.UNKNOWN
        if len(history) > 1:
            previous_status = HealthStatus(history[-2]['status'])

        # Trigger alert if status changed to warning/critical or recovered
        if (current_status in [HealthStatus.WARNING, HealthStatus.CRITICAL] and
                previous_status not in [HealthStatus.WARNING, HealthStatus.CRITICAL]) or \
                (current_status == HealthStatus.HEALTHY and
                 previous_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]):

            for callback in self.alert_callbacks:
                try:
                    callback(check_name, current_status, result)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def _check_system_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        if usage_percent > self.memory_critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critical: {usage_percent:.1f}%"
        elif usage_percent > self.memory_warning_threshold:
            status = HealthStatus.WARNING
            message = f"Memory usage high: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {usage_percent:.1f}%"

        return {
            'status': status.value,
            'message': message,
            'usage_percent': usage_percent,
            'available_gb': memory.available / (1024**3),
            'timestamp': time.time()
        }

    def _check_system_cpu(self) -> Dict[str, Any]:
        """Check system CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)

        if cpu_percent > self.cpu_critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critical: {cpu_percent:.1f}%"
        elif cpu_percent > self.cpu_warning_threshold:
            status = HealthStatus.WARNING
            message = f"CPU usage high: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"

        return {
            'status': status.value,
            'message': message,
            'usage_percent': cpu_percent,
            'timestamp': time.time()
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        disk = psutil.disk_usage(self.disk_check_path)
        usage_percent = disk.percent

        if usage_percent > self.disk_critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"Disk space critical: {usage_percent:.1f}%"
        elif usage_percent > self.disk_warning_threshold:
            status = HealthStatus.WARNING
            message = f"Disk space low: {usage_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk space normal: {usage_percent:.1f}%"

        return {
            'status': status.value,
            'message': message,
            'usage_percent': usage_percent,
            'free_gb': disk.free / (1024**3),
            'timestamp': time.time()
        }

    def _get_single_health_status(self, check_name: str) -> Dict[str, Any]:
        """
        Get health status for a specific check.

        Args:
            check_name: Name of the health check

        Returns:
            Dictionary containing check status information
        """
        if check_name not in self.health_checks:
            return {'error': f'Health check {check_name} not found'}

        check = self.health_checks[check_name]
        current_time = time.time()

        # Perform check if it's due
        if current_time - check.last_check >= check.interval:
            self.perform_health_check(check_name)

        return {
            'name': check_name,
            'status': check.last_result or {'status': HealthStatus.UNKNOWN.value},
            'last_check': check.last_check,
            'next_check': check.last_check + check.interval,
            'history': self.status_history[check_name][-5:]  # Last 5 results
        }

    def _get_all_health_statuses(self) -> Dict[str, Any]:
        """
        Get health status for all checks and calculate overall status.

        Returns:
            Dictionary containing overall status and all check statuses
        """
        all_statuses = {}
        overall_status = HealthStatus.HEALTHY

        for name in self.health_checks:
            status_info = self.get_health_status(name)
            all_statuses[name] = status_info

            # Determine overall status
            overall_status = self._calculate_overall_status(overall_status, status_info)

        return {
            'overall_status': overall_status.value,
            'checks': all_statuses,
            'timestamp': time.time()
        }

    def _calculate_overall_status(self, current_overall: HealthStatus, status_info: Dict) -> HealthStatus:
        """
        Calculate the overall health status based on individual check status.

        Args:
            current_overall: Current overall status
            status_info: Status information for a single check

        Returns:
            Updated overall health status
        """
        check_status = status_info['status']['status']

        if check_status == HealthStatus.CRITICAL.value:
            return HealthStatus.CRITICAL
        elif (check_status == HealthStatus.WARNING.value and
              current_overall == HealthStatus.HEALTHY):
            return HealthStatus.WARNING

        return current_overall
