"""
Tests for ScaffoldGenerator grouping and embedding logic.
"""
import unittest
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np

from src.cubo.processing.scaffold import ScaffoldGenerator


class TestScaffoldGenerator(unittest.TestCase):
    """Test ScaffoldGenerator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock enricher
        self.mock_enricher = MagicMock()
        self.mock_enricher.enrich_chunks.return_value = [
            {
                'text': 'Chunk 1 about AI',
                'summary': 'AI introduction',
                'keywords': ['AI', 'machine learning'],
                'category': 'technology',
                'consistency_score': 4.5
            },
            {
                'text': 'Chunk 2 about AI applications',
                'summary': 'AI applications in healthcare',
                'keywords': ['AI', 'healthcare'],
                'category': 'technology',
                'consistency_score': 4.2
            },
            {
                'text': 'Chunk 3 about finance',
                'summary': 'Financial markets overview',
                'keywords': ['finance', 'markets'],
                'category': 'finance',
                'consistency_score': 4.0
            }
        ]

        # Mock embedding generator
        self.mock_embedding_gen = MagicMock()
        self.mock_embedding_gen.generate_embeddings.return_value = [
            np.random.rand(384).tolist(),  # Scaffold 1 embedding
            np.random.rand(384).tolist(),  # Scaffold 2 embedding
        ]

        self.generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=2,
            similarity_threshold=0.75
        )

    def test_generate_scaffolds_basic(self):
        """Test basic scaffold generation."""
        chunks_df = pd.DataFrame({
            'chunk_id': ['c1', 'c2', 'c3'],
            'text': [
                'Chunk 1 about AI',
                'Chunk 2 about AI applications',
                'Chunk 3 about finance'
            ]
        })

        result = self.generator.generate_scaffolds(chunks_df)

        # Check result structure
        self.assertIn('scaffolds_df', result)
        self.assertIn('mapping', result)
        self.assertIn('scaffold_embeddings', result)

        # Verify scaffolds DataFrame
        scaffolds_df = result['scaffolds_df']
        self.assertIsInstance(scaffolds_df, pd.DataFrame)
        self.assertIn('scaffold_id', scaffolds_df.columns)
        self.assertIn('summary', scaffolds_df.columns)  # Changed from 'scaffold_text'
        self.assertIn('chunk_ids', scaffolds_df.columns)

        # Verify all chunks are mapped
        mapping = result['mapping']
        # mapping is scaffold_id -> chunk_ids, so we need to extract all chunk_ids
        all_mapped_chunks = []
        for chunk_ids in mapping.values():
            all_mapped_chunks.extend(chunk_ids)
        self.assertEqual(set(all_mapped_chunks), {'c1', 'c2', 'c3'})

    def test_scaffold_grouping_logic(self):
        """Test that chunks are grouped correctly by similarity."""
        chunks_df = pd.DataFrame({
            'chunk_id': ['c1', 'c2', 'c3'],
            'text': [
                'AI and machine learning',
                'AI applications',
                'Financial markets'
            ]
        })

        result = self.generator.generate_scaffolds(chunks_df)

        # Should create scaffolds (at least 1)
        scaffolds_df = result['scaffolds_df']
        self.assertGreater(len(scaffolds_df), 0)

        # Verify that each scaffold doesn't exceed scaffold_size chunks
        for _, row in scaffolds_df.iterrows():
            chunk_ids = row['chunk_ids']
            self.assertLessEqual(len(chunk_ids), self.generator.scaffold_size)

    def test_merge_summaries(self):
        """Test summary merging logic."""
        chunks = [
            {'summary': 'First summary.'},
            {'summary': 'Second summary.'},
            {'summary': 'Third summary.'}
        ]

        merged = self.generator._merge_summaries(chunks)

        self.assertIsInstance(merged, str)
        self.assertIn('First summary', merged)
        self.assertIn('Second summary', merged)
        self.assertIn('Third summary', merged)

    def test_generate_scaffold_id(self):
        """Test scaffold ID generation."""
        scaffold_id = self.generator._generate_scaffold_id(0, "AI summary")

        self.assertIsInstance(scaffold_id, str)
        self.assertTrue(scaffold_id.startswith('scaffold_'))

        # IDs should be deterministic for same inputs
        scaffold_id2 = self.generator._generate_scaffold_id(0, "AI summary")
        self.assertEqual(scaffold_id, scaffold_id2)

        # IDs should differ for different inputs
        scaffold_id3 = self.generator._generate_scaffold_id(1, "AI summary")
        self.assertNotEqual(scaffold_id, scaffold_id3)

    def test_empty_dataframe(self):
        """Test handling of empty input DataFrame."""
        empty_df = pd.DataFrame(columns=['chunk_id', 'text'])

        result = self.generator.generate_scaffolds(empty_df)

        # Should return empty structures
        self.assertEqual(len(result['scaffolds_df']), 0)
        self.assertEqual(len(result['mapping']), 0)
        self.assertEqual(len(result['scaffold_embeddings']), 0)

    def test_save_and_load_scaffolds(self):
        """Test scaffold persistence."""
        import tempfile
        from pathlib import Path

        chunks_df = pd.DataFrame({
            'chunk_id': ['c1', 'c2'],
            'text': ['Chunk 1', 'Chunk 2']
        })

        result = self.generator.generate_scaffolds(chunks_df)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            self.generator.save_scaffolds(result, output_path)

            # Verify files exist
            self.assertTrue((output_path / 'scaffold_metadata.parquet').exists())
            self.assertTrue((output_path / 'scaffold_mapping.json').exists())
            # Note: manifest.json is not created by current implementation

            # Load
            loaded_result = self.generator.load_scaffolds(output_path)

            # Verify loaded data matches original
            self.assertIn('scaffolds_df', loaded_result)
            self.assertIn('mapping', loaded_result)
            self.assertEqual(
                set(loaded_result['mapping'].keys()),
                set(result['mapping'].keys())
            )


if __name__ == '__main__':
    unittest.main()
