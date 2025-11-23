"""
Tests for MemoryStore.
"""
import unittest
from src.cubo.storage.memory_store import InMemoryCollection

class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryCollection()

    def test_initialization(self):
        self.assertIsInstance(self.store, InMemoryCollection)

    # Add more tests here
