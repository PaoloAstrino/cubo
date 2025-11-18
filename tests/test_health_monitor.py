"""
Tests for HealthMonitor
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.cubo.monitoring.health_monitor import HealthMonitor, HealthStatus, HealthCheck


class TestHealthMonitor:
    """Test cases for HealthMonitor class."""

    def test_initialization(self):
        """Test HealthMonitor initialization."""
        hm = HealthMonitor()

        # Check default health checks are added
        assert 'system_memory' in hm.health_checks
        assert 'system_cpu' in hm.health_checks
        assert 'disk_space' in hm.health_checks

        # Check initial state
        assert len(hm.alert_callbacks) == 0
        assert len(hm.status_history) == 3  # One for each default check

    def test_add_health_check(self):
        """Test adding custom health checks."""
        hm = HealthMonitor()

        def custom_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Custom check passed',
                'timestamp': time.time()
            }

        hm.add_health_check('custom_check', custom_check, interval=10.0, timeout=5.0)

        assert 'custom_check' in hm.health_checks
        check = hm.health_checks['custom_check']
        assert check.interval == 10.0
        assert check.timeout == 5.0
        assert 'custom_check' in hm.status_history

    def test_perform_health_check_success(self):
        """Test successful health check execution."""
        hm = HealthMonitor()

        def success_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'All good',
                'custom_metric': 42,
                'timestamp': time.time()
            }

        hm.add_health_check('test_check', success_check)

        result = hm.perform_health_check('test_check')

        assert result['status'] == HealthStatus.HEALTHY.value
        assert result['message'] == 'All good'
        assert result['custom_metric'] == 42

        # Check history is updated
        assert len(hm.status_history['test_check']) == 1

    def test_perform_health_check_failure(self):
        """Test health check failure handling."""
        hm = HealthMonitor()

        def failing_check():
            raise ValueError("Check failed")

        hm.add_health_check('failing_check', failing_check)

        result = hm.perform_health_check('failing_check')

        assert result['status'] == HealthStatus.CRITICAL.value
        assert 'Check failed' in result['message']
        assert 'error' in result

    def test_perform_health_check_timeout(self):
        """Test health check timeout."""
        hm = HealthMonitor()

        def slow_check():
            time.sleep(2)
            return {'status': HealthStatus.HEALTHY.value, 'message': 'Slow check'}

        hm.add_health_check('slow_check', slow_check, timeout=0.5)

        result = hm.perform_health_check('slow_check')

        assert result['status'] == HealthStatus.CRITICAL.value
        assert 'timed out' in result['message']

    def test_perform_health_check_unknown(self):
        """Test requesting unknown health check."""
        hm = HealthMonitor()

        result = hm.perform_health_check('unknown_check')

        assert result['status'] == HealthStatus.UNKNOWN.value
        assert 'not found' in result['message']

    def test_get_health_status_single_check(self):
        """Test getting status for a single health check."""
        hm = HealthMonitor()

        def test_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Test check',
                'timestamp': time.time()
            }

        hm.add_health_check('single_test', test_check)

        status = hm.get_health_status('single_test')

        assert status['name'] == 'single_test'
        assert 'status' in status
        assert 'last_check' in status
        assert 'next_check' in status
        assert 'history' in status

    def test_get_health_status_all_checks(self):
        """Test getting status for all health checks."""
        hm = HealthMonitor()

        # Mock system checks to return healthy status
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk:

            mock_memory.return_value = MagicMock(percent=50.0, available=8*1024**3)
            mock_cpu.return_value = 30.0
            mock_disk.return_value = MagicMock(percent=50.0, free=100*1024**3)

            # Add a test check
            def test_check():
                return {
                    'status': HealthStatus.WARNING.value,
                    'message': 'Test warning',
                    'timestamp': time.time()
                }

            hm.add_health_check('warning_check', test_check)

            status = hm.get_health_status()

            assert 'overall_status' in status
            assert 'checks' in status
            assert 'timestamp' in status

            # Should have overall warning status due to warning check
            assert status['overall_status'] == HealthStatus.WARNING.value

            # Check all default checks are present
            assert 'system_memory' in status['checks']
            assert 'system_cpu' in status['checks']
            assert 'disk_space' in status['checks']
            assert 'warning_check' in status['checks']

    def test_overall_status_determination(self):
        """Test overall status calculation logic."""
        hm = HealthMonitor()
        
        # Clear default checks for isolated testing
        hm.health_checks.clear()
        hm.status_history.clear()

        # Test healthy overall
        def healthy_check():
            return {'status': HealthStatus.HEALTHY.value, 'message': 'OK', 'timestamp': time.time()}

        hm.add_health_check('healthy', healthy_check)
        status = hm.get_health_status()
        assert status['overall_status'] == HealthStatus.HEALTHY.value

        # Test warning overall
        def warning_check():
            return {'status': HealthStatus.WARNING.value, 'message': 'Warning', 'timestamp': time.time()}

        hm.add_health_check('warning', warning_check)
        status = hm.get_health_status()
        assert status['overall_status'] == HealthStatus.WARNING.value

        # Test critical overall (takes precedence)
        def critical_check():
            return {'status': HealthStatus.CRITICAL.value, 'message': 'Critical', 'timestamp': time.time()}

        hm.add_health_check('critical', critical_check)
        status = hm.get_health_status()
        assert status['overall_status'] == HealthStatus.CRITICAL.value

    def test_run_all_checks(self):
        """Test running all health checks."""
        hm = HealthMonitor()

        def test_check():
            return {'status': HealthStatus.HEALTHY.value, 'message': 'Test', 'timestamp': time.time()}

        hm.add_health_check('test_run_all', test_check)

        results = hm.run_all_checks()

        # Should have all checks including the new one
        assert len(results) >= 4  # 3 default + 1 custom
        assert 'system_memory' in results
        assert 'test_run_all' in results

    def test_alert_callbacks(self):
        """Test alert callback system."""
        hm = HealthMonitor()

        alerts_received = []

        def alert_callback(check_name, status, result):
            alerts_received.append((check_name, status.value, result['message']))

        hm.add_alert_callback(alert_callback)

        # Add a check that will trigger an alert
        def warning_check():
            return {
                'status': HealthStatus.WARNING.value,
                'message': 'Warning condition',
                'timestamp': time.time()
            }

        hm.add_health_check('alert_test', warning_check)

        # First run - should trigger alert
        hm.perform_health_check('alert_test')
        assert len(alerts_received) == 1
        assert alerts_received[0][0] == 'alert_test'
        assert alerts_received[0][1] == HealthStatus.WARNING.value

        # Second run - should not trigger alert (status didn't change)
        alerts_received.clear()
        hm.perform_health_check('alert_test')
        assert len(alerts_received) == 0

        # Change to healthy - should trigger recovery alert
        def healthy_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Back to healthy',
                'timestamp': time.time()
            }

        hm.health_checks['alert_test'].check_function = healthy_check
        hm.perform_health_check('alert_test')
        assert len(alerts_received) == 1
        assert alerts_received[0][1] == HealthStatus.HEALTHY.value

    def test_alert_callback_failure(self):
        """Test handling of failing alert callbacks."""
        hm = HealthMonitor()

        def failing_callback(check_name, status, result):
            raise ValueError("Callback failed")

        def good_callback(check_name, status, result):
            pass  # This should still work

        hm.add_alert_callback(failing_callback)
        hm.add_alert_callback(good_callback)

        def warning_check():
            return {
                'status': HealthStatus.WARNING.value,
                'message': 'Warning',
                'timestamp': time.time()
            }

        hm.add_health_check('callback_test', warning_check)

        # Should not raise exception even though first callback fails
        hm.perform_health_check('callback_test')

    def test_system_memory_check(self):
        """Test system memory health check."""
        hm = HealthMonitor()

        with patch('psutil.virtual_memory') as mock_memory:
            # Test healthy memory
            mock_memory.return_value = MagicMock(percent=50.0, available=8*1024**3)
            result = hm._check_system_memory()

            assert result['status'] == HealthStatus.HEALTHY.value
            assert 'normal' in result['message']
            assert result['usage_percent'] == 50.0

            # Test warning memory
            mock_memory.return_value = MagicMock(percent=85.0, available=4*1024**3)
            result = hm._check_system_memory()

            assert result['status'] == HealthStatus.WARNING.value
            assert 'high' in result['message']

            # Test critical memory
            mock_memory.return_value = MagicMock(percent=95.0, available=1*1024**3)
            result = hm._check_system_memory()

            assert result['status'] == HealthStatus.CRITICAL.value
            assert 'critical' in result['message']

    def test_system_cpu_check(self):
        """Test system CPU health check."""
        hm = HealthMonitor()

        with patch('psutil.cpu_percent') as mock_cpu:
            # Test healthy CPU
            mock_cpu.return_value = 30.0
            result = hm._check_system_cpu()

            assert result['status'] == HealthStatus.HEALTHY.value
            assert 'normal' in result['message']
            assert result['usage_percent'] == 30.0

            # Test warning CPU
            mock_cpu.return_value = 90.0
            result = hm._check_system_cpu()

            assert result['status'] == HealthStatus.WARNING.value
            assert 'high' in result['message']

            # Test critical CPU
            mock_cpu.return_value = 98.0
            result = hm._check_system_cpu()

            assert result['status'] == HealthStatus.CRITICAL.value
            assert 'critical' in result['message']

    def test_disk_space_check(self):
        """Test disk space health check."""
        hm = HealthMonitor()

        with patch('psutil.disk_usage') as mock_disk:
            # Test healthy disk
            mock_disk.return_value = MagicMock(percent=50.0, free=100*1024**3)
            result = hm._check_disk_space()

            assert result['status'] == HealthStatus.HEALTHY.value
            assert 'normal' in result['message']
            assert result['usage_percent'] == 50.0

            # Test warning disk
            mock_disk.return_value = MagicMock(percent=92.0, free=20*1024**3)
            result = hm._check_disk_space()

            assert result['status'] == HealthStatus.WARNING.value
            assert 'low' in result['message']

            # Test critical disk
            mock_disk.return_value = MagicMock(percent=97.0, free=5*1024**3)
            result = hm._check_disk_space()

            assert result['status'] == HealthStatus.CRITICAL.value
            assert 'critical' in result['message']

    def test_history_management(self):
        """Test health check history management."""
        hm = HealthMonitor()

        def test_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': f'Check {time.time()}',
                'timestamp': time.time()
            }

        hm.add_health_check('history_test', test_check)

        # Run check multiple times
        for i in range(12):  # More than the limit of 10
            hm.perform_health_check('history_test')

        # Should only keep last 10 results
        assert len(hm.status_history['history_test']) == 10

        # Check history in status
        status = hm.get_health_status('history_test')
        assert len(status['history']) == 5  # get_health_status returns last 5

    def test_check_scheduling(self):
        """Test that checks are only run when due."""
        hm = HealthMonitor()

        def test_check():
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Scheduled check',
                'timestamp': time.time()
            }

        hm.add_health_check('schedule_test', test_check, interval=10.0)

        # First call should run the check
        status1 = hm.get_health_status('schedule_test')
        first_check_time = status1['last_check']

        # Second call immediately after should not run again
        time.sleep(0.1)
        status2 = hm.get_health_status('schedule_test')
        second_check_time = status2['last_check']

        assert first_check_time == second_check_time

        # Manually expire the check
        hm.health_checks['schedule_test'].last_check = time.time() - 15.0

        # Third call should run again
        status3 = hm.get_health_status('schedule_test')
        third_check_time = status3['last_check']

        assert third_check_time > second_check_time