"""
Tests for WindowReplacementPostProcessor.
"""

import unittest

from src.cubo.processing.postprocessor import WindowReplacementPostProcessor


class TestWindowReplacementPostProcessor(unittest.TestCase):
    """Test WindowReplacementPostProcessor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = WindowReplacementPostProcessor(target_metadata_key="window")

    def test_replace_with_window_context(self):
        """Test that sentences are replaced with window context."""
        retrieval_results = [
            {
                "document": "This is a single sentence.",
                "metadata": {
                    "window": "Previous sentence. This is a single sentence. Next sentence.",
                    "source": "doc1.txt",
                },
                "score": 0.95,
            }
        ]

        processed = self.processor.postprocess_results(retrieval_results)

        self.assertEqual(len(processed), 1)
        self.assertEqual(
            processed[0]["document"], "Previous sentence. This is a single sentence. Next sentence."
        )
        # Metadata should remain unchanged
        self.assertEqual(processed[0]["metadata"]["source"], "doc1.txt")
        self.assertEqual(processed[0]["score"], 0.95)

    def test_no_window_in_metadata(self):
        """Test that results without window metadata are unchanged."""
        retrieval_results = [
            {
                "document": "This is a sentence without window.",
                "metadata": {"source": "doc2.txt"},
                "score": 0.85,
            }
        ]

        processed = self.processor.postprocess_results(retrieval_results)

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["document"], "This is a sentence without window.")

    def test_empty_window_context(self):
        """Test that empty window context doesn't replace document."""
        retrieval_results = [
            {
                "document": "Original sentence.",
                "metadata": {"window": "   ", "source": "doc3.txt"},
                "score": 0.75,
            }
        ]

        processed = self.processor.postprocess_results(retrieval_results)

        self.assertEqual(processed[0]["document"], "Original sentence.")

    def test_multiple_results(self):
        """Test processing multiple results with mixed window availability."""
        retrieval_results = [
            {
                "document": "Sentence with window.",
                "metadata": {"window": "Context for sentence with window."},
                "score": 0.95,
            },
            {
                "document": "Sentence without window.",
                "metadata": {"source": "doc.txt"},
                "score": 0.85,
            },
            {
                "document": "Another with window.",
                "metadata": {"window": "Full context for another with window."},
                "score": 0.80,
            },
        ]

        processed = self.processor.postprocess_results(retrieval_results)

        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0]["document"], "Context for sentence with window.")
        self.assertEqual(processed[1]["document"], "Sentence without window.")
        self.assertEqual(processed[2]["document"], "Full context for another with window.")

    def test_custom_metadata_key(self):
        """Test using a custom metadata key for window context."""
        processor = WindowReplacementPostProcessor(target_metadata_key="context")

        retrieval_results = [
            {
                "document": "Original.",
                "metadata": {"context": "Custom context window."},
                "score": 0.90,
            }
        ]

        processed = processor.postprocess_results(retrieval_results)
        self.assertEqual(processed[0]["document"], "Custom context window.")

    def test_empty_results_list(self):
        """Test that empty results list doesn't cause errors."""
        processed = self.processor.postprocess_results([])
        self.assertEqual(processed, [])


if __name__ == "__main__":
    unittest.main()
