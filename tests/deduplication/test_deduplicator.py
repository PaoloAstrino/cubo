"""
Tests for Deduplicator.
"""

import pytest
pytest.importorskip("torch")

import unittest

from cubo.deduplication.deduplicator import Deduplicator


class TestDeduplicator(unittest.TestCase):
    def setUp(self):
        self.deduplicator = Deduplicator()

    def test_initialization(self):
        self.assertIsInstance(self.deduplicator, Deduplicator)

    # Add more tests here based on Deduplicator methods
