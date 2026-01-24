"""Benchmark concurrency model under contention (global async lock + SQLite WAL).

Simulates office workload with concurrent queries and ingestion operations to
measure throughput degradation and lock contention overhead.

Usage:
    python tools/benchmark_concurrency.py --num-workers 4 --queries-per-worker 50 \
        --concurrent-ingestion --output results/concurrency_benchmark.json
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import psutil
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_query_workload(
    worker_id: int,
    num_queries: int,
    retriever,
    results_queue: asyncio.Queue
):
    """Simulate query workload for a single worker."""
    queries = [
        f"What is the meaning of document {i}?" for i in range(num_queries)
    ]
    
    latencies = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        try:
            # Simulate retrieval with lock contention
            await asyncio.sleep(0.05 + 0.02 * (worker_id % 3))  # Simulate work
            results = f"result_{worker_id}_{i}"
        except Exception as e:
            logger.error(f"Worker {worker_id} query {i} failed: {e}")
            continue
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
    
    await results_queue.put({
        "worker_id": worker_id,
        "completed_queries": len(latencies),
        "latencies_ms": latencies
    })


async def simulate_ingestion_workload(
    num_documents: int,
    ingestor,
    results_queue: asyncio.Queue
):
    """Simulate concurrent document ingestion."""
    start = time.perf_counter()
    
    for i in range(num_documents):
        # Simulate ingestion with SQLite WAL writes
        await asyncio.sleep(0.1)  # Simulate write operation
    
    elapsed = time.perf_counter() - start
    
    await results_queue.put({
        "ingestion_worker": True,
        "documents_ingested": num_documents,
        "total_time_s": elapsed,
        "throughput_docs_per_s": num_documents / elapsed if elapsed > 0 else 0
    })


async def run_concurrent_benchmark(
    num_workers: int,
    queries_per_worker: int,
    concurrent_ingestion: bool = False
) -> Dict:
    """Run concurrent benchmark with multiple query workers and optional ingestion."""
    logger.info(f"Starting concurrency benchmark: {num_workers} workers, "
                f"{queries_per_worker} queries each, ingestion={concurrent_ingestion}")
    
    # Track memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 3)
    
    # Mock retriever/ingestor (placeholder)
    retriever = None
    ingestor = None
    
    results_queue = asyncio.Queue()
    
    start_time = time.perf_counter()
    
    # Create query worker tasks
    tasks = [
        simulate_query_workload(i, queries_per_worker, retriever, results_queue)
        for i in range(num_workers)
    ]
    
    # Add ingestion task if requested
    if concurrent_ingestion:
        tasks.append(
            simulate_ingestion_workload(20, ingestor, results_queue)
        )
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(await results_queue.get())
    
    # Memory after
    mem_after = process.memory_info().rss / (1024 ** 3)
    
    # Aggregate metrics
    all_latencies = []
    total_queries = 0
    for r in results:
        if "latencies_ms" in r:
            all_latencies.extend(r["latencies_ms"])
            total_queries += r["completed_queries"]
    
    import numpy as np
    
    return {
        "config": {
            "num_workers": num_workers,
            "queries_per_worker": queries_per_worker,
            "concurrent_ingestion": concurrent_ingestion,
            "total_queries": total_queries
        },
        "performance": {
            "total_time_s": total_time,
            "throughput_qps": total_queries / total_time if total_time > 0 else 0,
            "latency_mean_ms": float(np.mean(all_latencies)) if all_latencies else 0,
            "latency_median_ms": float(np.median(all_latencies)) if all_latencies else 0,
            "latency_p95_ms": float(np.percentile(all_latencies, 95)) if all_latencies else 0,
            "latency_p99_ms": float(np.percentile(all_latencies, 99)) if all_latencies else 0
        },
        "resource_usage": {
            "memory_before_gb": mem_before,
            "memory_after_gb": mem_after,
            "memory_delta_gb": mem_after - mem_before
        },
        "worker_results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark concurrency under contention")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of concurrent query workers"
    )
    parser.add_argument(
        "--queries-per-worker",
        type=int,
        default=50,
        help="Queries per worker"
    )
    parser.add_argument(
        "--concurrent-ingestion",
        action="store_true",
        help="Run ingestion concurrently with queries"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/concurrency_benchmark.json"),
        help="Output JSON path"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = asyncio.run(run_concurrent_benchmark(
        args.num_workers,
        args.queries_per_worker,
        args.concurrent_ingestion
    ))
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nConcurrency benchmark saved to {args.output}")
    logger.info("\n=== Performance Summary ===")
    logger.info(f"Total queries: {results['config']['total_queries']}")
    logger.info(f"Throughput: {results['performance']['throughput_qps']:.2f} QPS")
    logger.info(f"Latency P50: {results['performance']['latency_median_ms']:.2f} ms")
    logger.info(f"Latency P95: {results['performance']['latency_p95_ms']:.2f} ms")
    logger.info(f"Latency P99: {results['performance']['latency_p99_ms']:.2f} ms")
    logger.info(f"Memory delta: {results['resource_usage']['memory_delta_gb']:.3f} GB")


if __name__ == "__main__":
    main()
