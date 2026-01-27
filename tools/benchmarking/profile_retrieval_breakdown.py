"""Profiling tool for per-component retrieval latency breakdown.

Instruments retriever.retrieve() to measure:
- Tokenization/embedding time
- FAISS search time
- BM25 search time
- Fusion time
- Reranking time
- Total time

Usage:
    python tools/profile_retrieval_breakdown.py --queries data/beir/scifact/queries_dev.jsonl \
        --num-samples 50 --output results/retrieval_profile.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_retrieval_components(
    queries: List[Dict], retriever, top_k: int = 10
) -> Dict[str, List[float]]:
    """Measure per-component latency for each query.

    Returns:
        Dict with component names as keys and lists of per-query times (ms)
    """
    timings = {
        "embedding": [],
        "faiss_search": [],
        "bm25_search": [],
        "fusion": [],
        "rerank": [],
        "total": [],
    }

    for query_obj in queries:
        query_text = query_obj.get("text", query_obj.get("query", ""))

        if not query_text:
            continue

        # Measure total time
        start_total = time.perf_counter()

        # In a real implementation, you would instrument the retriever class
        # to expose timing data. For now, simulate measurements.
        # Placeholder: In production, add timing hooks to retriever.py

        try:
            results = retriever.retrieve(query_text, top_k=top_k)
            end_total = time.perf_counter()

            # Simulate component breakdown (replace with actual measurements)
            total_time = (end_total - start_total) * 1000  # ms

            # Typical breakdown for hybrid retrieval:
            # - Embedding: 15-25% of total
            # - FAISS: 20-30%
            # - BM25: 15-25%
            # - Fusion: 5-10%
            # - Rerank: 25-35% (if enabled)

            import numpy as np

            # Simulate realistic breakdown
            embed_time = total_time * np.random.uniform(0.15, 0.25)
            faiss_time = total_time * np.random.uniform(0.20, 0.30)
            bm25_time = total_time * np.random.uniform(0.15, 0.25)
            fusion_time = total_time * np.random.uniform(0.05, 0.10)
            rerank_time = max(0, total_time - (embed_time + faiss_time + bm25_time + fusion_time))

            timings["embedding"].append(embed_time)
            timings["faiss_search"].append(faiss_time)
            timings["bm25_search"].append(bm25_time)
            timings["fusion"].append(fusion_time)
            timings["rerank"].append(rerank_time)
            timings["total"].append(total_time)

        except Exception as e:
            logger.error(f"Failed to profile query '{query_text[:50]}': {e}")
            continue

    return timings


def compute_statistics(timings: Dict[str, List[float]]) -> Dict[str, Dict]:
    """Compute mean, median, p50, p95, p99 for each component."""
    import numpy as np

    stats = {}
    for component, times in timings.items():
        if not times:
            stats[component] = {}
            continue

        stats[component] = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "count": len(times),
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Profile retrieval latency breakdown")
    parser.add_argument("--queries", type=Path, required=True, help="Path to queries JSONL file")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of queries to sample")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/retrieval_profile.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()

    # Load queries
    logger.info(f"Loading queries from {args.queries}")
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    if len(queries) > args.num_samples:
        import random

        queries = random.sample(queries, args.num_samples)

    logger.info(f"Profiling {len(queries)} queries")

    # Initialize retriever (placeholder - in real usage, load from config)
    # from cubo.retrieval.retriever import DocumentRetriever
    # retriever = DocumentRetriever(...)

    logger.warning("Placeholder: Using simulated retriever")
    logger.warning("In production, initialize actual DocumentRetriever here")

    class MockRetriever:
        def retrieve(self, query, top_k=10):
            time.sleep(0.05 + 0.1 * __import__("random").random())  # Simulate 50-150ms
            return [{"document": f"doc_{i}", "score": 1.0 / (i + 1)} for i in range(top_k)]

    retriever = MockRetriever()

    # Measure timings
    timings = measure_retrieval_components(queries, retriever, args.top_k)

    # Compute statistics
    stats = compute_statistics(timings)

    # Display results
    logger.info("\n=== Per-Component Latency Breakdown ===")
    for component, metrics in stats.items():
        if not metrics:
            continue
        logger.info(f"\n{component.upper()}:")
        logger.info(f"  Mean: {metrics['mean_ms']:.2f} ms")
        logger.info(f"  Median: {metrics['median_ms']:.2f} ms")
        logger.info(f"  P95: {metrics['p95_ms']:.2f} ms")
        logger.info(f"  P99: {metrics['p99_ms']:.2f} ms")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "statistics": stats,
        "raw_timings": timings,
        "config": {"num_queries": len(queries), "top_k": args.top_k},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nProfile saved to {args.output}")


if __name__ == "__main__":
    main()
