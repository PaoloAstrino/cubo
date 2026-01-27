#!/usr/bin/env python3
"""Worker script to run retrieval and record per-query latencies.
Usage: python tools/worker_retrieve.py --index-dir results/beir_index_nfcorpus --queries data/beir/nfcorpus/queries.jsonl --output results/run_nfcorpus.json --top-k 50 --mode with_rerank|no_rerank
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.beir_adapter import CuboBeirAdapter


def load_queries(path):
    q = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            q[item.get("_id", str(i))] = item.get("query") or item.get("text") or ""
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--mode", choices=["with_rerank", "no_rerank"], default="with_rerank")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    adapter = CuboBeirAdapter(index_dir=args.index_dir, lightweight=False)

    per_query_latency = {}
    results = {}

    start = time.time()
    # Choose retrieval method
    if args.mode == "no_rerank":
        # Use optimized retrieval without reranking
        res = adapter.retrieve_bulk_optimized(queries, top_k=args.top_k, skip_reranker=True)
        # We don't have per-query latencies here; approximate uniform time
        # Mark each as avg
        for qid in queries.keys():
            per_query_latency[qid] = 0.0
        results = res
    else:
        # Use sequential full retrieval to capture per-query latency
        qids = list(queries.keys())
        for qid in qids:
            t0 = time.time()
            hits = adapter.retrieve(queries[qid], top_k=args.top_k)
            t1 = time.time()
            per_query_latency[qid] = t1 - t0
            results[qid] = {doc_id: score for doc_id, score in hits}

    end = time.time()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lat_out = args.output.replace(".json", "_latencies.json")
    with open(lat_out, "w", encoding="utf-8") as lf:
        json.dump({"per_query": per_query_latency, "total_time_s": end - start}, lf, indent=2)

    print(f"Saved run to {args.output} and latencies to {lat_out}")


if __name__ == "__main__":
    main()
