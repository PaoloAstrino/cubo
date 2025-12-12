"""
Tests for HierarchicalChunker in CustomAutoMerging.
"""

import pytest
pytest.importorskip("torch")

import unittest

from cubo.deduplication.custom_auto_merging import AutoMergingChunker


class TestHierarchicalChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = AutoMergingChunker(chunk_sizes=[100, 50])

    def test_initialization(self):
        self.assertIsInstance(self.chunker, AutoMergingChunker)

    def test_create_hierarchical_chunks(self):
        text = "This is a test document. " * 20
        chunks = self.chunker.create_hierarchical_chunks(text, "test.txt")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        # Check basic structure
        self.assertIn("id", chunks[0])
        self.assertIn("text", chunks[0])
        self.assertIn("level", chunks[0])


if __name__ == "__main__":
    unittest.main()
