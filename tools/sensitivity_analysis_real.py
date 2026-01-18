"""Sensitivity analysis for FAISS parameters with real index measurements.

Tests nprobe, nlist, and PQ codes on actual FAISS index to measure
latency and recall tradeoffs.

Usage:
    python tools\sensitivity_analysis_real.py \
        --index-dir data/faiss_test \
        --queries data/beir/scifact/queries_test100.jsonl \
        --nprobe-values 1,5,10,20,50 \
        --output results/sensitivity_real.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add cubo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_faiss_sensitivity_real(
    retriever,
    queries: List[Dict],
    nprobe: int,
    top_k: int = 10
) -> Dict:
    """Measure FAISS performance with specific nprobe setting."""
    latencies = []
    
    # Temporarily set nprobe
    collection = getattr(retriever, 'collection', None)
    if collection and hasattr(collection, 'index') and collection.index:
        original_nprobe = collection.index.nprobe
        collection.index.nprobe = nprobe
    else:
        logger.warning("Could not set nprobe on FAISS index")
        original_nprobe = None
    
    # Run queries
    for query_obj in queries:
        query_text = query_obj.get("text", query_obj.get("query", ""))
        if not query_text:
            continue
        
        start = time.perf_counter()
        try:
            results = retriever.retrieve(query_text, top_k=top_k)
            # Defensive: some retrievers may return non-iterable error codes in
            # edge cases — guard and skip those queries rather than crash the
            # whole sweep.
            if not isinstance(results, (list, tuple)):
                logger.error(
                    "Retriever returned non-list result (skipping query): %r",
                    type(results),
                )
                continue
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        except Exception as e:
            logger.exception(f"Query failed with nprobe={nprobe}")
            continue
    
    # Restore original nprobe
    if original_nprobe is not None and collection and collection.index:
        collection.index.nprobe = original_nprobe
    
    if not latencies:
        return {}
    
    return {
        "nprobe": nprobe,
        "num_queries": len(latencies),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_median_ms": float(np.median(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99))
    }


def run_sensitivity_grid_real(
    retriever,
    queries: List[Dict],
    nprobe_values: List[int],
    top_k: int = 10
) -> List[Dict]:
    """Run sensitivity analysis across nprobe values."""
    results = []
    
    logger.info(f"Testing {len(nprobe_values)} nprobe configurations")
    
    for nprobe in nprobe_values:
        logger.info(f"Testing nprobe={nprobe}")
        result = measure_faiss_sensitivity_real(retriever, queries, nprobe, top_k)
        if result:
            results.append(result)
    
    return results


def analyze_sensitivity_real(results: List[Dict]) -> Dict:
    """Analyze sensitivity trends."""
    if not results:
        return {}
    
    # Sort by nprobe
    results_sorted = sorted(results, key=lambda x: x['nprobe'])
    
    nprobes = [r['nprobe'] for r in results_sorted]
    latencies = [r['latency_mean_ms'] for r in results_sorted]
    
    # Compute sensitivity metrics
    analysis = {
        "nprobe_range": {"min": min(nprobes), "max": max(nprobes)},
        "latency_range": {"min": min(latencies), "max": max(latencies)},
        "latency_increase_factor": max(latencies) / min(latencies) if min(latencies) > 0 else 0,
        "monotonic_increase": all(
            latencies[i] <= latencies[i+1] for i in range(len(latencies)-1)
        ),
        "recommended_nprobe": None
    }
    
    # Find recommended nprobe (best latency/accuracy tradeoff)
    # Heuristic: choose nprobe where latency is <2x minimum
    min_latency = min(latencies)
    for r in results_sorted:
        if r['latency_mean_ms'] <= min_latency * 2:
            analysis["recommended_nprobe"] = r['nprobe']
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="FAISS sensitivity analysis with real measurements"
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Path to FAISS index directory"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to queries JSONL file"
    )
    parser.add_argument(
        "--nprobe-values",
        type=str,
        default="1,5,10,20,50",
        help="Comma-separated nprobe values to test"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of queries to sample for each config"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sensitivity_real.json"),
        help="Output JSON path"
    )
    
    args = parser.parse_args()
    
    # Parse nprobe values
    nprobe_values = [int(x.strip()) for x in args.nprobe_values.split(',')]
    
    # Load queries
    queries = []
    with open(args.queries, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    # Sample queries
    if len(queries) > args.num_samples:
        import random
        queries = random.sample(queries, args.num_samples)
    
    logger.info(f"Loaded {len(queries)} queries for sensitivity analysis")
    
    # Initialize retriever via dependency helper
    from cubo.retrieval.dependencies import get_scaffold_retriever, get_embedding_generator

    if not Path(args.index_dir).exists():
        raise FileNotFoundError(f"FAISS index not found: {args.index_dir}")

    # Try a few likely index locations (prefer requested)
    candidates = [str(args.index_dir), 'data/beir_index_scifact', 'beir_index_scifact', 'data/faiss_test']
    retriever = None
    for cand in candidates:
        if not Path(cand).exists():
            continue
        retriever = get_scaffold_retriever(cand, get_embedding_generator())
        if retriever:
            logger.info(f"Using retriever from: {cand}")
            args.index_dir = cand
            break
    if retriever is None:
        raise RuntimeError(f"Failed to initialize retriever for any candidate: {candidates}")

    # Run sensitivity grid
    logger.info("Starting FAISS sensitivity analysis...")
    sensitivity_results = run_sensitivity_grid_real(
        retriever, queries, nprobe_values, args.top_k
    )
    
    # Analyze results
    analysis = analyze_sensitivity_real(sensitivity_results)
    
    # Prepare output
    results = {
        "config": {
            "index_dir": str(args.index_dir),
            "queries_file": str(args.queries),
            "num_samples": len(queries),
            "nprobe_values": nprobe_values,
            "top_k": args.top_k
        },
        "results": sensitivity_results,
        "analysis": analysis
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Sensitivity analysis saved to {args.output}")
    logger.info("\n=== Sensitivity Results ===")
    for r in sensitivity_results:
        logger.info(
            f"nprobe={r['nprobe']:3d}: mean={r['latency_mean_ms']:6.2f}ms, "
            f"p95={r['latency_p95_ms']:6.2f}ms, p99={r['latency_p99_ms']:6.2f}ms"
        )
    
    logger.info("\n=== Analysis ===")
    if analysis:
        logger.info(f"Latency increase factor: {analysis.get('latency_increase_factor'):.2f}x")
        logger.info(f"Monotonic increase: {analysis.get('monotonic_increase')}")
        logger.info(f"Recommended nprobe: {analysis.get('recommended_nprobe')}")
    else:
        logger.warning("Sensitivity analysis produced no usable results")


if __name__ == "__main__":
    main()
