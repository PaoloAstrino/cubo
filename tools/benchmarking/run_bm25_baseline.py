#!/usr/bin/env python
"""Run BM25 baseline queries and measure performance metrics.

Computes recall@k, NDCG@k, peak memory, and query latency.

Usage:
    python tools/run_bm25_baseline.py \\
        --index-dir data/beir_index_bm25_scifact \\
        --queries data/beir/scifact/queries.jsonl \\
        --qrels data/beir/scifact/qrels/test.txt \\
        --top-k 100 \\
        --num-samples 200 \\
        --output results/baselines/scifact/bm25_full.json
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
import psutil


def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file in standard format: qid docid relevance."""
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = rel
    return qrels


def run_pyserini_query(
    index_dir: Path, query_id: str, query_text: str, top_k: int = 100
) -> List[Tuple[str, float]]:
    """Run single query against Pyserini index and return ranked results."""

    try:
        from pyserini.search.lucene import LuceneSearcher

        if not hasattr(run_pyserini_query, "searcher"):
            print(f"Initializing Pyserini searcher...")
            run_pyserini_query.searcher = LuceneSearcher(str(index_dir))

        searcher = run_pyserini_query.searcher
        hits = searcher.search(query_text, k=top_k)

        results = [(hit.docid, hit.score) for hit in hits]
        return results

    except ImportError:
        return []


def run_fallback_bm25_query(
    index_file: Path, query_text: str, top_k: int = 100
) -> List[Tuple[str, float]]:
    """Fallback: Simple BM25 scoring without Pyserini."""

    # Load index if not cached
    if not hasattr(run_fallback_bm25_query, "index"):
        print(f"Loading BM25 index from {index_file}...")
        with open(index_file) as f:
            run_fallback_bm25_query.index = json.load(f)

    index = run_fallback_bm25_query.index

    # Tokenize query
    tokens = [t.lower() for t in query_text.split() if t.isalnum()]

    # Simple BM25 scoring (k1=1.5, b=0.75, epsilon=0.25)
    k1, b = 1.5, 0.75

    doc_scores = {}
    for doc_id in index["doc_lengths"].keys():
        doc_id = int(doc_id)
        score = 0.0
        doc_len = index["doc_lengths"].get(str(doc_id), 1)

        for token in tokens:
            if token in index["inverted_index"]:
                idf = np.log(
                    (len(index["doc_id_map"]) - len(index["inverted_index"][token]) + 0.5)
                    / (len(index["inverted_index"][token]) + 0.5)
                    + 1.0
                )

                tf = index["doc_term_counts"].get(str(doc_id), {}).get(token, 0)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + idf * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (doc_len / 1000))
                )

    # Sort and return top-k
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = [
        (index["doc_id_map"].get(str(doc_id), str(doc_id)), score) for doc_id, score in ranked
    ]
    return results


def calculate_recall_at_k(
    retrieved_ids: List[str], relevant_docs: Dict[str, int], k: int = 10
) -> float:
    """Calculate recall@k: fraction of relevant docs in top-k retrieved."""

    relevant_at_k = sum(
        1 for doc_id in retrieved_ids[:k] if doc_id in relevant_docs and relevant_docs[doc_id] > 0
    )
    total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)

    return relevant_at_k / total_relevant if total_relevant > 0 else 0.0


def calculate_ndcg_at_k(
    retrieved_ids: List[str], relevant_docs: Dict[str, int], k: int = 10
) -> float:
    """Calculate NDCG@k."""

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        relevance = relevant_docs.get(doc_id, 0)
        if relevance > 0:
            dcg += relevance / np.log2(i + 2)  # log2(i+2) for 1-indexed ranking

    # Calculate ideal DCG (all relevant docs first)
    relevant_list = sorted([rel for rel in relevant_docs.values() if rel > 0], reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevant_list[:k]))

    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run BM25 baseline")
    parser.add_argument("--index-dir", type=Path, required=True, help="BM25 index directory")
    parser.add_argument("--queries", type=Path, required=True, help="Queries JSONL file")
    parser.add_argument("--qrels", type=Path, help="Qrels file for evaluation")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k results to retrieve")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit queries (for testing)")
    parser.add_argument("--output", type=Path, default=Path("results/bm25_baseline.json"))
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")

    args = parser.parse_args()

    print("=" * 70)
    print("BM25 BASELINE QUERIES")
    print("=" * 70)

    # Load queries
    print(f"\nLoading queries from {args.queries}...")
    queries = {}
    with open(args.queries) as f:
        for line in f:
            doc = json.loads(line)
            qid = doc.get("_id", doc.get("id"))
            text = doc.get("text", doc.get("query"))
            queries[qid] = text

    if args.num_samples:
        query_ids = list(queries.keys())[: args.num_samples]
        queries = {qid: queries[qid] for qid in query_ids}

    print(f"[OK] Loaded {len(queries)} queries")

    # Load qrels if available
    qrels = {}
    if args.qrels and args.qrels.exists():
        qrels = load_qrels(args.qrels)
        print(f"[OK] Loaded qrels for {len(qrels)} queries")

    # Check for index
    if not args.index_dir.exists():
        print(f"[FAIL] Index not found at {args.index_dir}")
        print(f"   Run index_bm25_pyserini.py first")
        return 1

    # Memory tracking
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)
    peak_memory = initial_memory

    # Run queries
    print(f"\nRunning {len(queries)} queries...")
    latencies = []
    all_results = {}

    start_time = time.time()

    for i, (qid, query_text) in enumerate(queries.items()):
        q_start = time.time()

        # Try Pyserini first, fallback to pure Python
        try:
            results = run_pyserini_query(args.index_dir, qid, query_text, args.top_k)
        except:
            index_file = args.index_dir / "bm25_index.json"
            results = run_fallback_bm25_query(index_file, query_text, args.top_k)

        latency_ms = (time.time() - q_start) * 1000
        latencies.append(latency_ms)

        # Store results
        all_results[qid] = {
            "results": [(doc_id, float(score)) for doc_id, score in results],
            "latency_ms": latency_ms,
        }

        # Track memory
        if args.profile_memory:
            current_memory = process.memory_info().rss / (1024**3)
            peak_memory = max(peak_memory, current_memory)

        if (i + 1) % max(10, len(queries) // 10) == 0:
            avg_lat = np.mean(latencies[-100:]) if len(latencies) >= 100 else np.mean(latencies)
            print(
                f"  [{i+1}/{len(queries)}] Avg latency: {avg_lat:.1f}ms, Peak memory: {peak_memory:.1f}GB"
            )

    total_time = time.time() - start_time

    # Compute metrics
    print(
        f"\n[OK] Queries completed in {total_time:.1f}s ({total_time/len(queries):.2f}s per query)"
    )

    recall_10_scores = []
    ndcg_10_scores = []

    for qid, qdata in all_results.items():
        if qid in qrels:
            retrieved_ids = [doc_id for doc_id, _ in qdata["results"]]
            recall_10 = calculate_recall_at_k(retrieved_ids, qrels[qid], k=10)
            ndcg_10 = calculate_ndcg_at_k(retrieved_ids, qrels[qid], k=10)

            recall_10_scores.append(recall_10)
            ndcg_10_scores.append(ndcg_10)

    # Compile results
    results = {
        "model": "BM25",
        "dataset": "unknown",
        "num_queries": len(queries),
        "performance": {
            "total_time_s": total_time,
            "throughput_qps": len(queries) / total_time,
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_median_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_min_ms": float(np.min(latencies)),
            "latency_max_ms": float(np.max(latencies)),
        },
        "evaluation": {
            "recall@10": float(np.mean(recall_10_scores)) if recall_10_scores else 0.0,
            "ndcg@10": float(np.mean(ndcg_10_scores)) if ndcg_10_scores else 0.0,
        },
        "resource_usage": {
            "memory_before_gb": initial_memory,
            "memory_after_gb": process.memory_info().rss / (1024**3),
            "peak_memory_gb": peak_memory,
        },
    }

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {args.output}")
    print(f"{'=' * 70}")
    print(f"Recall@10:       {results['evaluation']['recall@10']:.3f}")
    print(f"NDCG@10:         {results['evaluation']['ndcg@10']:.3f}")
    print(f"Latency p95:     {results['performance']['latency_p95_ms']:.0f} ms")
    print(f"Throughput:      {results['performance']['throughput_qps']:.2f} QPS")
    print(f"Peak Memory:     {peak_memory:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
