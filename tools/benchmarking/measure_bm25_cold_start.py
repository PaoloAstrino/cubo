#!/usr/bin/env python3
"""
Measure BM25 cold-start vs warm-cache latencies.

This script measures the latency difference between:
1. Cold-start: First query after OS cache flush (disk I/O penalty)
2. Warm-cache: Subsequent queries with hot inverted lists in memory

Usage:
    python measure_bm25_cold_start.py --corpus scifact --queries 50 --runs 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cubo.retrieval.bm25_python_store import BM25PythonStore


def flush_os_cache():
    """Flush OS file system cache (requires admin/sudo on most systems)."""
    import platform

    system = platform.system()
    if system == "Linux":
        os.system("sudo sync; sudo echo 3 > /proc/sys/vm/drop_caches")
        time.sleep(0.5)
    elif system == "Darwin":  # macOS
        os.system("sudo purge")
        time.sleep(0.5)
    elif system == "Windows":
        # On Windows, use RAMMap from Sysinternals (requires download)
        # Alternative: restart search service
        print("⚠️  Windows: Cannot automatically flush cache. Run:")
        print(
            "   1. Download RAMMap from https://learn.microsoft.com/en-us/sysinternals/downloads/rammap"
        )
        print("   2. Click 'Empty → Empty Working Sets' before cold-start test")
        print("   Otherwise, latencies will reflect warm cache only.")
    else:
        print(f"⚠️  Unknown OS: {system}. Cache flush not supported.")


def measure_bm25_latency(
    bm25_store: BM25PythonStore,
    queries: List[str],
    top_k: int = 10,
    is_cold_start: bool = False,
    num_runs: int = 1,
) -> Dict[str, float]:
    """
    Measure BM25 search latencies.

    Args:
        bm25_store: BM25PythonStore instance
        queries: List of query strings
        top_k: Number of results to return
        is_cold_start: If True, assume cache is empty (only test first query)
        num_runs: Number of repeated runs

    Returns:
        Dictionary with p50, p95, p99, mean, and min latencies in ms
    """
    latencies = []

    for run in range(num_runs):
        if is_cold_start and run > 0:
            # For cold-start, only measure the first query
            break

        # Flush cache before cold-start run
        if is_cold_start and run == 0:
            print("  Flushing OS cache (requires admin)...")
            flush_os_cache()
            time.sleep(1)  # Give OS time to stabilize

        for query in queries:
            start = time.perf_counter()
            results = bm25_store.search(query, top_k=top_k)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

    # Compute percentiles
    latencies = sorted(latencies)
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "samples": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure BM25 cold-start vs warm-cache latencies")
    parser.add_argument(
        "--corpus",
        type=str,
        default="scifact",
        choices=["scifact", "fiqa", "arguana", "nfcorpus"],
        help="BEIR corpus to use",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Path to existing BM25 index. If not provided, will use default location.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=50,
        help="Number of queries to measure",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of warm-cache runs",
    )
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="Skip cold-start test (require manual cache flush, slow)",
    )

    args = parser.parse_args()

    # Generate sample queries
    print(f"Generating {args.queries} sample queries...")
    sample_queries = [
        "what is machine learning",
        "neural networks deep learning",
        "transformer models NLP",
        "BERT language understanding",
        "attention mechanism",
        "information retrieval",
        "semantic search",
        "vector embeddings",
        "dense retrieval",
        "sparse retrieval BM25",
    ]
    # Cycle through sample queries if needed
    query_list = (sample_queries * (args.queries // len(sample_queries) + 1))[: args.queries]
    print(f"Using {len(query_list)} queries")

    # Determine index directory
    if args.index_dir is None:
        # Try common locations
        possible_paths = [
            f"data/beir_index_bm25_{args.corpus}",
            f"data/{args.corpus}/bm25_index",
            f"data/bm25_{args.corpus}",
        ]
        args.index_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                args.index_dir = path
                break

    if not args.index_dir or not os.path.exists(args.index_dir):
        print(f"❌ BM25 index not found")
        print(f"   Searched in common locations (e.g., data/beir_index_bm25_{args.corpus})")
        print(f"   Available BEIR BM25 indices:")
        import glob

        for idx in sorted(glob.glob("data/beir_index_bm25_*")):
            print(f"     - {idx}")
        sys.exit(1)

    # Load BM25 store
    print(f"Loading BM25 index from {args.index_dir}...")
    bm25_store = BM25PythonStore(index_dir=args.index_dir)

    results = {}

    # Cold-start test (requires cache flush)
    if not args.skip_cold:
        print("\n🔴 COLD-START TEST (disk I/O, cache empty)")
        print("   WARNING: This requires admin privileges to flush OS cache.")
        print("   On Windows, manually use RAMMap → Empty Working Sets before running.")
        print("   Measuring first query only...")

        try:
            cold_latencies = measure_bm25_latency(
                bm25_store,
                query_list,
                top_k=10,
                is_cold_start=True,
                num_runs=1,
            )
            results["cold_start"] = cold_latencies
            print(f"   p50: {cold_latencies['p50']:.1f} ms")
            print(f"   p95: {cold_latencies['p95']:.1f} ms")
            print(f"   p99: {cold_latencies['p99']:.1f} ms")
        except Exception as e:
            print(f"   ⚠️  Cold-start failed: {e}")
            print("   Skipping cold-start test.")

    # Warm-cache test
    print("\n🟢 WARM-CACHE TEST (hot inverted lists in memory)")
    print(f"   Running {args.runs} iterations with {len(query_list)} queries each...")

    warm_latencies = measure_bm25_latency(
        bm25_store,
        query_list,
        top_k=10,
        is_cold_start=False,
        num_runs=args.runs,
    )
    results["warm_cache"] = warm_latencies

    print(f"   p50: {warm_latencies['p50']:.1f} ms")
    print(f"   p95: {warm_latencies['p95']:.1f} ms")
    print(f"   p99: {warm_latencies['p99']:.1f} ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if "cold_start" in results:
        cold = results["cold_start"]
        warm = results["warm_cache"]
        print(f"\nCold-Start (1 query, empty cache):")
        print(f"  p50:  {cold['p50']:6.1f} ms")
        print(f"  p95:  {cold['p95']:6.1f} ms")
        print(f"  p99:  {cold['p99']:6.1f} ms")

        print(f"\nWarm-Cache ({warm['samples']} queries, hot lists):")
        print(f"  p50:  {warm['p50']:6.1f} ms")
        print(f"  p95:  {warm['p95']:6.1f} ms")
        print(f"  p99:  {warm['p99']:6.1f} ms")

        print(f"\nPenalty (Cold - Warm):")
        print(f"  p50:  {cold['p50'] - warm['p50']:6.1f} ms")
        print(f"  p95:  {cold['p95'] - warm['p95']:6.1f} ms")
        print(f"  p99:  {cold['p99'] - warm['p99']:6.1f} ms")
    else:
        warm = results["warm_cache"]
        print(f"\nWarm-Cache ({warm['samples']} queries, hot lists):")
        print(f"  p50:  {warm['p50']:6.1f} ms")
        print(f"  p95:  {warm['p95']:6.1f} ms")
        print(f"  p99:  {warm['p99']:6.1f} ms")

    # Save results
    output_file = f"bm25_latency_{args.corpus}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")

    # Recommendations for paper
    print("\n" + "=" * 60)
    print("FOR PAPER")
    print("=" * 60)

    if "cold_start" in results:
        cold = results["cold_start"]
        warm = results["warm_cache"]
        print(f"\nUse this text in Table 6 footnote:")
        print(f"\n  BM25 latencies measured with warm index cache; cold-start")
        print(
            f"  latencies (empty OS cache, disk I/O) are {cold['p50']:.0f}–{cold['p95']:.0f} ms p50–p95."
        )
        print(f"  In production, BM25 achieves <{warm['p50']:.0f} ms after warm-up.")


if __name__ == "__main__":
    main()
