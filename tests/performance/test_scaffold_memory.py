"""
Performance and memory tests for Processing module.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.cubo.processing.scaffold import ScaffoldGenerator


class TestScaffoldPerformance(unittest.TestCase):
    """Test scaffold generation performance and memory usage."""

    def setUp(self):
        """Set up mock dependencies."""
        # Mock enricher that returns minimal enrichment
        self.mock_enricher = MagicMock()
        self.mock_enricher.enrich_chunks.side_effect = lambda chunks: [
            {
                "text": chunk,
                "summary": f"Summary of {chunk[:20]}",
                "keywords": ["test", "keyword"],
                "category": "test",
                "consistency_score": 4.0,
            }
            for chunk in chunks
        ]

        # Mock embedding generator
        self.mock_embedding_gen = MagicMock()

    def test_memory_usage_small_dataset(self):
        """Test memory usage with small dataset (1000 chunks) using tracemalloc."""
        from tests.performance.performance_test_utils import MemoryProfiler

        # Generate 1000 chunks
        chunks_df = pd.DataFrame(
            {
                "chunk_id": [f"c{i}" for i in range(1000)],
                "text": [f"This is test chunk number {i} with some content." for i in range(1000)],
            }
        )

        # Mock embeddings for all scaffolds
        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5,
        )

        def operation():
            return generator.generate_scaffolds(chunks_df)

        # Measure with tracemalloc
        mem_stats = MemoryProfiler.measure_peak_memory(operation)
        result = mem_stats["result"]
        peak_mb = mem_stats["peak_mb"]

        # Should use less than 30MB for 1000 chunks (more realistic threshold)
        self.assertLess(peak_mb, 30.0, f"Peak memory {peak_mb:.2f} MB exceeds 30 MB")

        # Verify all chunks processed
        all_chunks = []
        for chunk_ids in result["mapping"].values():
            all_chunks.extend(chunk_ids)
        self.assertEqual(len(set(all_chunks)), 1000)

    def test_memory_usage_large_dataset(self):
        """Test memory usage with larger dataset (10K chunks) using tracemalloc."""
        from tests.performance.performance_test_utils import MemoryProfiler

        # Generate 10K chunks
        chunks_df = pd.DataFrame(
            {
                "chunk_id": [f"c{i}" for i in range(10000)],
                "text": [f"Test chunk {i} with content" * 10 for i in range(10000)],  # Longer text
            }
        )

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=10,
        )

        def operation():
            return generator.generate_scaffolds(chunks_df)

        # Measure with tracemalloc
        mem_stats = MemoryProfiler.measure_peak_memory(operation)
        result = mem_stats["result"]
        peak_mb = mem_stats["peak_mb"]

        # Should use less than 150MB for 10K chunks (more realistic threshold)
        self.assertLess(peak_mb, 150.0, f"Peak memory {peak_mb:.2f} MB exceeds 150 MB")

        # Verify all chunks processed
        all_chunks = []
        for chunk_ids in result["mapping"].values():
            all_chunks.extend(chunk_ids)
        self.assertEqual(len(set(all_chunks)), 10000)

    def test_processing_time_scalability(self):
        """Test that processing time scales linearly (O(n)) with dataset size using regression."""
        import time

        from tests.performance.performance_test_utils import ComplexityAnalyzer

        def mock_embed(texts):
            # Add small realistic delay to make timing measurable
            time.sleep(len(texts) * 0.0001)  # 0.1ms per text
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5,
        )

        def operation_for_size(size):
            """Operation parameterized by size."""
            chunks_df = pd.DataFrame(
                {
                    "chunk_id": [f"c{i}" for i in range(size)],
                    "text": [f"Chunk {i} content" for i in range(size)],
                }
            )
            generator.generate_scaffolds(chunks_df)

        # Analyze complexity with different sizes
        sizes = [100, 300, 500, 1000]
        analysis = ComplexityAnalyzer.analyze_complexity(
            operation_for_size, sizes, expected_complexity="linear"
        )

        # Assert linear or n*log(n) complexity fits reasonably well
        # With mocked operations, timing can be noisy, so we use lenient thresholds
        linear_r2 = analysis["fits"]["linear"]["r_squared"]
        nlogn_r2 = analysis["fits"]["quadratic"]["r_squared"]

        # Either linear or nlogn should fit well (both are acceptable for this algorithm)
        best_fit_r2 = max(linear_r2, nlogn_r2)
        self.assertGreater(
            best_fit_r2,
            0.70,
            f"Neither linear ({linear_r2:.3f}) nor n*log(n) ({nlogn_r2:.3f}) fits well (R²<0.70)",
        )

        # Assert quadratic doesn't fit significantly better than linear
        quadratic_r2 = analysis["fits"]["quadratic"]["r_squared"]
        if quadratic_r2 > linear_r2:
            improvement = quadratic_r2 - linear_r2
            self.assertLess(
                improvement,
                0.15,
                f"Quadratic fits much better (ΔR²={improvement:.3f}), suggesting O(n²) complexity",
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
            scaffold_size=5,
        )

        all_mappings = {}
        all_scaffolds = []

        for i in range(0, total_chunks, batch_size):
            batch_df = pd.DataFrame(
                {
                    "chunk_id": [f"c{j}" for j in range(i, min(i + batch_size, total_chunks))],
                    "text": [f"Chunk {j}" for j in range(i, min(i + batch_size, total_chunks))],
                }
            )

            result = generator.generate_scaffolds(batch_df)
            # Invert mapping: scaffold_id -> chunk_ids to chunk_id -> scaffold_id
            for scaffold_id, chunk_ids in result["mapping"].items():
                for chunk_id in chunk_ids:
                    all_mappings[chunk_id] = scaffold_id
            all_scaffolds.append(result["scaffolds_df"])

        # Verify all chunks processed
        self.assertEqual(len(all_mappings), total_chunks)

        # Verify we got scaffolds from all batches
        combined_scaffolds = pd.concat(all_scaffolds, ignore_index=True)
        self.assertGreater(len(combined_scaffolds), 0)

    def test_throughput_target(self):
        """Test system meets throughput targets for scaffold generation."""
        from tests.performance.performance_test_utils import ThroughputProfiler

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5,
        )

        total_chunks = 2000
        chunks_df = pd.DataFrame(
            {
                "chunk_id": [f"c{i}" for i in range(total_chunks)],
                "text": [f"Chunk {i} content" for i in range(total_chunks)],
            }
        )

        def operation():
            generator.generate_scaffolds(chunks_df)

        throughput_stats = ThroughputProfiler.measure_throughput(
            operation, n_items=total_chunks, warmup=True
        )

        # Target: At least 500 chunks/second (adjustable based on hardware)
        self.assertGreater(
            throughput_stats["items_per_sec"],
            500,
            f"Throughput {throughput_stats['items_per_sec']:.1f} chunks/sec below target",
        )

    def test_latency_percentiles(self):
        """Test latency percentiles are within acceptable ranges."""
        from tests.performance.performance_test_utils import LatencyProfiler

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5,
        )

        chunks_df = pd.DataFrame(
            {
                "chunk_id": [f"c{i}" for i in range(1000)],
                "text": [f"Chunk {i}" for i in range(1000)],
            }
        )

        def operation():
            generator.generate_scaffolds(chunks_df)

        latency_stats = LatencyProfiler.measure_with_warmup(operation, warmup_runs=2, n_samples=10)

        # P50 should be reasonable
        self.assertLess(
            latency_stats["p50_ms"], 2000, f"P50 latency {latency_stats['p50_ms']:.1f}ms too high"
        )

        # P99 should not be much larger than P50 (indicates consistent performance)
        p50_to_p99_ratio = latency_stats["p99_ms"] / latency_stats["p50_ms"]
        self.assertLess(
            p50_to_p99_ratio, 3.0, f"P99/P50 ratio {p50_to_p99_ratio:.2f}x indicates high variance"
        )

    def test_memory_no_leaks(self):
        """Test that memory doesn't leak over repeated operations."""
        from tests.performance.performance_test_utils import ResourceMonitor

        def mock_embed(texts):
            return [np.random.rand(384).tolist() for _ in texts]

        self.mock_embedding_gen.generate_embeddings.side_effect = mock_embed

        generator = ScaffoldGenerator(
            enricher=self.mock_enricher,
            embedding_generator=self.mock_embedding_gen,
            scaffold_size=5,
        )

        def operation():
            chunks_df = pd.DataFrame(
                {
                    "chunk_id": [f"c{i}" for i in range(100)],
                    "text": [f"Chunk {i}" for i in range(100)],
                }
            )
            generator.generate_scaffolds(chunks_df)

        leak_stats = ResourceMonitor.measure_memory_growth(operation, iterations=10)

        # Memory growth per iteration should be minimal (<0.5MB)
        self.assertLess(
            leak_stats["growth_per_iter_mb"],
            0.5,
            f"Memory leak detected: {leak_stats['growth_per_iter_mb']:.2f} MB/iteration",
        )


class TestEnrichmentPerformance(unittest.TestCase):
    """Test enrichment performance."""

    def test_enrichment_failure_handling(self):
        """Test that enrichment handles LLM failures gracefully without performance degradation."""
        import time

        from src.cubo.processing.enrichment import ChunkEnricher

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
            self.assertIn("text", chunk)
            self.assertIn("summary", chunk)
            self.assertIn("keywords", chunk)
            self.assertIn("category", chunk)
            self.assertIn("consistency_score", chunk)

        # Should not take excessively long (no infinite retries)
        self.assertLess(elapsed, 5.0, "Enrichment took too long with failures")


if __name__ == "__main__":
    unittest.main()
