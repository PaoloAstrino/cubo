"""
Tests for MetadataManager.
"""

import unittest

from cubo.storage.metadata_manager import MetadataManager


class TestMetadataManager(unittest.TestCase):
    def setUp(self):
        self.manager = MetadataManager()

    def test_initialization(self):
        self.assertIsInstance(self.manager, MetadataManager)

    # Add more tests here
