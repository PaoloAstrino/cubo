"""
Tests for HealthMonitor.
"""

import unittest

from cubo.monitoring.health_monitor import HealthMonitor


class TestHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = HealthMonitor()

    def test_initialization(self):
        self.assertIsInstance(self.monitor, HealthMonitor)

    # Add more tests here
