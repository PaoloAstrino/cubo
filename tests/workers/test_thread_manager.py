"""
Tests for ThreadManager.
"""
import unittest
from unittest.mock import MagicMock
from src.cubo.workers.thread_manager import ThreadManager

class TestThreadManager(unittest.TestCase):
    def setUp(self):
        self.manager = ThreadManager()

    def test_initialization(self):
        self.assertIsInstance(self.manager, ThreadManager)

    # Add more tests here
