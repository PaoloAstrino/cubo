#!/usr/bin/env python
"""Run e5-base-v2 queries on IVFPQ index and measure performance metrics.

Loads pre-built FAISS IVFPQ index and document embeddings, executes batch queries,
measures latency/throughput/memory, and outputs JSON results.

Usage:
    python tools/run_e5_baseline.py \\
        --index-dir data/beir_index_e5_scifact \\
        --queries data/beir/scifact/queries.jsonl \\
        --top-k 100 \\
        --num-samples 200 \\
        --output results/baselines/scifact/e5_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psutil
import os


def load_queries(queries_file: Path, limit: int = None) -> dict:
    """Load queries from JSONL file."""
    queries = {}
    with open(queries_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            qid = doc.get("_id", doc.get("id", str(i)))
            text = doc.get("text", doc.get("query", ""))
            queries[qid] = text
    return queries


def run_e5_queries(index_dir: Path, queries: dict, top_k: int = 100) -> List[Tuple[str, List, float]]:
    """Run queries against e5 IVFPQ index."""
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)
    
    # Load metadata and index
    metadata_file = index_dir / "e5_metadata.json"
    index_file = index_dir / "e5_ivfpq.index"
    doc_ids_file = index_dir / "doc_ids.json"
    
    if not all([metadata_file.exists(), index_file.exists(), doc_ids_file.exists()]):
        print(f"[ERROR] Missing index files in {index_dir}")
        return []
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    with open(doc_ids_file) as f:
        doc_ids = json.load(f)
    
    print(f"Loading e5 model and index...")
    model = SentenceTransformer("intfloat/e5-base-v2")
    index = faiss.read_index(str(index_file))
    
    print(f"Index: {metadata['num_vectors']} vectors, dimension={metadata['dimension']}")
    print(f"\nRunning {len(queries)} queries...")
    
    results_list = []
    latencies = []
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 3)
    peak_memory = initial_memory
    
    start_time = time.time()
    
    for i, (qid, query_text) in enumerate(queries.items()):
        q_start = time.time()
        
        # Encode query
        query_embedding = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype(np.float32)
        
        # Search index
        distances, indices = index.search(query_embedding, top_k)
        
        latency_ms = (time.time() - q_start) * 1000
        latencies.append(latency_ms)
        
        # Map indices to doc IDs
        doc_results = [(doc_ids[int(idx)], float(1 - dist)) for idx, dist in zip(indices[0], distances[0])]
        
        results_list.append((qid, doc_results, latency_ms))
        
        # Track memory
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = max(peak_memory, current_memory)
        
        if (i + 1) % max(10, len(queries) // 10) == 0:
            avg_lat = np.mean(latencies[-100:]) if len(latencies) >= 100 else np.mean(latencies)
            print(f"  [{i+1}/{len(queries)}] Avg latency: {avg_lat:.1f}ms, Peak memory: {peak_memory:.1f}GB")
    
    total_time = time.time() - start_time
    print(f"\n[OK] Completed {len(queries)} queries in {total_time:.1f}s")
    
    return results_list, latencies, initial_memory, peak_memory


def main():
    parser = argparse.ArgumentParser(description="Run e5-base-v2 queries")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory with FAISS index")
    parser.add_argument("--queries", type=Path, required=True, help="Queries JSONL file")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k results")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit queries")
    parser.add_argument("--output", type=Path, default=Path("results/e5_baseline.json"))
    
    args = parser.parse_args()
    
    if not args.index_dir.exists():
        print(f"[ERROR] Index directory not found: {args.index_dir}")
        return 1
    
    if not args.queries.exists():
        print(f"[ERROR] Queries file not found: {args.queries}")
        return 1
    
    print("=" * 70)
    print("E5-BASE-V2 BASELINE QUERIES")
    print("=" * 70)
    
    # Load queries
    queries = load_queries(args.queries, args.num_samples)
    print(f"[OK] Loaded {len(queries)} queries")
    
    # Run queries
    results_list, latencies, initial_memory, peak_memory = run_e5_queries(args.index_dir, queries, args.top_k)
    
    if not results_list:
        print("[ERROR] Query execution failed")
        return 1
    
    # Compile results JSON
    total_time = sum(lat for _, _, lat in results_list) / 1000  # Convert ms to s
    
    results = {
        "model": "e5-base-v2",
        "dataset": "unknown",
        "num_queries": len(queries),
        "performance": {
            "total_time_s": sum(lat for _, _, lat in results_list) / 1000,
            "throughput_qps": len(queries) / (sum(lat for _, _, lat in results_list) / 1000),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_min_ms": float(np.min(latencies)),
            "latency_max_ms": float(np.max(latencies))
        },
        "evaluation": {
            "recall@10": 0.0,  # Would need qrels
            "ndcg@10": 0.0
        },
        "resource_usage": {
            "memory_before_gb": initial_memory,
            "memory_after_gb": peak_memory,
            "peak_memory_gb": peak_memory
        }
    }
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {args.output}")
    print(f"{'=' * 70}")
    print(f"Throughput:      {results['performance']['throughput_qps']:.2f} QPS")
    print(f"Latency p50:     {results['performance']['latency_median_ms']:.0f} ms")
    print(f"Latency p95:     {results['performance']['latency_p95_ms']:.0f} ms")
    print(f"Peak Memory:     {peak_memory:.2f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
