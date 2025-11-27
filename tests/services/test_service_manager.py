"""
Tests for ServiceManager.
"""

import unittest

from src.cubo.services.service_manager import ServiceManager, get_service_manager


class TestServiceManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton if necessary or test instance directly
        self.manager = ServiceManager()

    def test_initialization(self):
        self.assertIsInstance(self.manager, ServiceManager)

    def test_singleton_access(self):
        manager1 = get_service_manager()
        manager2 = get_service_manager()
        self.assertIs(manager1, manager2)

    # Add more tests here
