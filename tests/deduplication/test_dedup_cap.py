"""Tests for deduplication with candidate cap."""

import unittest


class TestDeduplicatorCandidateCap(unittest.TestCase):
    """Tests for the Deduplicator candidate pair cap feature."""

    def test_no_cap_by_default(self):
        """Test that cap is None by default (unlimited)."""
        from unittest.mock import patch
        from cubo.deduplication.deduplicator import Deduplicator

        # Ensure config does not provide max_candidates to validate default behavior
        with patch("cubo.deduplication.deduplicator.config") as mock_config:
            mock_config.get.return_value = {}
            dedup = Deduplicator()
            self.assertIsNone(dedup.max_candidates)

    def test_explicit_cap(self):
        """Test setting cap explicitly."""
        from cubo.deduplication.deduplicator import Deduplicator

        dedup = Deduplicator(max_candidates=100)
        self.assertEqual(dedup.max_candidates, 100)

    def test_small_dataset_not_affected(self):
        """Test that small datasets work normally with cap."""
        from cubo.deduplication.deduplicator import Deduplicator

        dedup = Deduplicator(threshold=0.5, max_candidates=1000)

        documents = [
            {"doc_id": "doc1", "text": "hello world this is a test"},
            {"doc_id": "doc2", "text": "hello world this is also a test"},
            {"doc_id": "doc3", "text": "completely different content here"},
        ]

        canonical_map = dedup.deduplicate(documents)

        # Should work normally
        self.assertEqual(len(canonical_map), 3)

        # Stats should show we weren't capped
        stats = dedup.get_stats()
        self.assertFalse(stats["was_capped"])

    def test_cap_limits_pairs(self):
        """Test that cap actually limits candidate pairs."""
        from cubo.deduplication.deduplicator import Deduplicator

        # Create many similar documents that would generate many pairs
        dedup = Deduplicator(threshold=0.3, max_candidates=5)

        # Create documents with overlapping content
        base_words = "the quick brown fox jumps over the lazy dog"
        documents = [{"doc_id": f"doc{i}", "text": f"{base_words} extra{i}"} for i in range(20)]

        canonical_map = dedup.deduplicate(documents)

        # Should have processed something
        self.assertGreater(len(canonical_map), 0)

        # Stats should show we hit the cap
        stats = dedup.get_stats()
        self.assertLessEqual(stats["pairs_used"], 5)

    def test_get_stats(self):
        """Test get_stats returns expected fields."""
        from cubo.deduplication.deduplicator import Deduplicator

        dedup = Deduplicator(max_candidates=100)

        documents = [
            {"doc_id": "doc1", "text": "hello world"},
            {"doc_id": "doc2", "text": "goodbye world"},
        ]

        dedup.deduplicate(documents)

        stats = dedup.get_stats()

        self.assertIn("pairs_found", stats)
        self.assertIn("pairs_used", stats)
        self.assertIn("max_candidates", stats)
        self.assertIn("was_capped", stats)
        self.assertEqual(stats["max_candidates"], 100)

    def test_duplicate_detection_with_cap(self):
        """Test that duplicates are still found with reasonable cap."""
        from cubo.deduplication.deduplicator import Deduplicator

        dedup = Deduplicator(threshold=0.5, max_candidates=50)

        # Create some actual duplicates
        documents = [
            {"doc_id": "doc1", "text": "the quick brown fox jumps over the lazy dog"},
            {"doc_id": "doc2", "text": "the quick brown fox jumps over the lazy dog exactly"},
            {"doc_id": "doc3", "text": "a completely different document about cats"},
            {"doc_id": "doc4", "text": "another document about cats and dogs"},
        ]

        canonical_map = dedup.deduplicate(documents)

        # doc1 and doc2 should be in same cluster
        self.assertEqual(canonical_map["doc1"], canonical_map["doc2"])

        # doc3 should be its own canonical
        self.assertEqual(canonical_map["doc3"], "doc3")

    def test_prioritizes_high_match_docs(self):
        """Test that cap prioritizes docs with more potential matches."""
        from cubo.deduplication.deduplicator import Deduplicator

        dedup = Deduplicator(threshold=0.4, max_candidates=3)

        # Create a hub document that matches many others
        hub_text = "common words that appear everywhere in multiple documents"
        documents = [
            {"doc_id": "hub", "text": hub_text},
            {"doc_id": "spoke1", "text": hub_text + " extra1"},
            {"doc_id": "spoke2", "text": hub_text + " extra2"},
            {"doc_id": "spoke3", "text": hub_text + " extra3"},
            {"doc_id": "unrelated", "text": "xyz abc completely different"},
        ]

        canonical_map = dedup.deduplicate(documents)

        # Even with cap, hub should be connected to at least some spokes
        hub_canonical = canonical_map["hub"]
        connected_to_hub = sum(
            1 for doc_id in ["spoke1", "spoke2", "spoke3"] if canonical_map[doc_id] == hub_canonical
        )

        self.assertGreater(connected_to_hub, 0)


class TestDeduplicatorConfigIntegration(unittest.TestCase):
    """Tests for config-based candidate cap."""

    def test_reads_from_config(self):
        """Test that Deduplicator reads max_candidates from config."""
        from unittest.mock import patch

        from cubo.deduplication.deduplicator import Deduplicator

        # Mock config to return max_candidates
        with patch("cubo.deduplication.deduplicator.config") as mock_config:
            mock_config.get.return_value = {"max_candidates": 200}

            dedup = Deduplicator()

            self.assertEqual(dedup.max_candidates, 200)

    def test_explicit_overrides_config(self):
        """Test that explicit max_candidates overrides config."""
        from unittest.mock import patch

        from cubo.deduplication.deduplicator import Deduplicator

        with patch("cubo.deduplication.deduplicator.config") as mock_config:
            mock_config.get.return_value = {"max_candidates": 200}

            # Explicit value should override
            dedup = Deduplicator(max_candidates=50)

            self.assertEqual(dedup.max_candidates, 50)


if __name__ == "__main__":
    unittest.main()
