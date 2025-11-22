"""
Performance and memory tests for Processing module.
"""
import unittest
import psutil
import pandas as pd
from unittest.mock import MagicMock
import numpy as np

from src.cubo.processing.scaffold import ScaffoldGenerator


class TestScaffoldPerformance(unittest.TestCase):
    """Test scaffold generation performance and memory usage."""

    def setUp(self):
        """Set up mock dependencies."""
        # Mock enricher that returns minimal enrichment
        self.mock_enricher = MagicMock()
        self.mock_enricher.enrich_chunks.side_effect = lambda chunks: [
            {
                'text': chunk,
                'summary': f'Summary of {chunk[:20]}',
                'keywords': ['test', 'keyword'],
                'category': 'test',
                'consistency_score': 4.0
            }
            for chunk in chunks
        ]

        # Mock embedding generator
        self.mock_embedding_gen = MagicMock()
        
    def test_memory_usage_small_dataset(self):
        """Test memory usage with small dataset (1000 chunks)."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Generate 1000 chunks
        chunks_df = pd.DataFrame({
            'chunk_id': [f'c{i}' for i in range(1000)],
            'text': [f'This is test chunk number {i} with some content.' for i in range(1000)]
        })

        # Mock embeddings for all scaffolds
        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]
        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5
        )

        result = generator.generate_scaffolds(chunks_df)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        # Should use less than 50MB for 1000 chunks
        self.assertLess(mem_used, 50.0, f"Used {mem_used:.2f} MB for 1000 chunks")

        # Verify all chunks processed
        self.assertEqual(len(result['chunk_to_scaffold_mapping']), 1000)

    def test_memory_usage_large_dataset(self):
        """Test memory usage with larger dataset (10K chunks)."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Generate 10K chunks
        chunks_df = pd.DataFrame({
            'chunk_id': [f'c{i}' for i in range(10000)],
            'text': [f'Test chunk {i} with content' * 10 for i in range(10000)]  # Longer text
        })

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]
        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=10
        )

        result = generator.generate_scaffolds(chunks_df)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before

        # Should use less than 200MB for 10K chunks
        self.assertLess(mem_used, 200.0, f"Used {mem_used:.2f} MB for 10K chunks")

        # Verify all chunks processed
        self.assertEqual(len(result['chunk_to_scaffold_mapping']), 10000)

    def test_processing_time_scalability(self):
        """Test that processing time scales linearly with dataset size."""
        import time

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]
        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5
        )

        # Test with different sizes
        sizes_and_times = []

        for size in [100, 500, 1000]:
            chunks_df = pd.DataFrame({
                'chunk_id': [f'c{i}' for i in range(size)],
                'text': [f'Chunk {i} content' for i in range(size)]
            })

            start = time.time()
            generator.generate_scaffolds(chunks_df)
            elapsed = time.time() - start

            sizes_and_times.append((size, elapsed))

        # Check that processing time grows sub-quadratically
        # Ratio of (time2/time1) should be less than (size2/size1)^2
        _, time_100 = sizes_and_times[0]
        _, time_1000 = sizes_and_times[2]

        time_ratio = time_1000 / time_100 if time_100 > 0 else float('inf')
        size_ratio_squared = (1000 / 100) ** 2  # 100

        self.assertLess(
            time_ratio,
            size_ratio_squared,
            f"Processing time grew too fast: {time_ratio:.2f}x vs expected <{size_ratio_squared}x"
        )

    def test_batch_processing(self):
        """Test batch processing to handle very large datasets."""
        # Simulate very large dataset by processing in batches
        total_chunks = 5000
        batch_size = 1000

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]
        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5
        )

        all_mappings = {}
        all_scaffolds = []

        for i in range(0, total_chunks, batch_size):
            batch_df = pd.DataFrame({
                'chunk_id': [f'c{j}' for j in range(i, min(i + batch_size, total_chunks))],
                'text': [f'Chunk {j}' for j in range(i, min(i + batch_size, total_chunks))]
            })

            result = generator.generate_scaffolds(batch_df)
            all_mappings.update(result['chunk_to_scaffold_mapping'])
            all_scaffolds.append(result['scaffolds_df'])

        # Verify all chunks processed
        self.assertEqual(len(all_mappings), total_chunks)

        # Verify we got scaffolds from all batches
        combined_scaffolds = pd.concat(all_scaffolds, ignore_index=True)
        self.assertGreater(len(combined_scaffolds), 0)


class TestEnrichmentPerformance(unittest.TestCase):
    """Test enrichment performance."""

    def test_enrichment_failure_handling(self):
        """Test that enrichment handles LLM failures gracefully without performance degradation."""
        from src.cubo.processing.enrichment import ChunkEnricher
        import time

        # Mock LLM that fails 50% of the time
        mock_llm = MagicMock()
        call_count = [0]
        
        def failing_generate(prompt, context):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("Simulated LLM failure")
            return "Success response"
        
        mock_llm.generate_response.side_effect = failing_generate

        enricher = ChunkEnricher(llm_provider=mock_llm)

        chunks = [f"Test chunk {i}" for i in range(10)]

        start = time.time()
        enriched = enricher.enrich_chunks(chunks)
        elapsed = time.time() - start

        # All chunks should be processed despite failures
        self.assertEqual(len(enriched), 10)

        # Failed chunks should have empty/default values
        for chunk in enriched:
            self.assertIn('text', chunk)
            self.assertIn('summary', chunk)
            self.assertIn('keywords', chunk)
            self.assertIn('category', chunk)
            self.assertIn('consistency_score', chunk)

        # Should not take excessively long (no infinite retries)
        self.assertLess(elapsed, 5.0, "Enrichment took too long with failures")


if __name__ == '__main__':
    unittest.main()
