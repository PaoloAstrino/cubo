"""Run SPLADE queries on sparse vector index and measure performance.

Loads pre-built SPLADE index (sparse vectors) and executes batch queries,
measures latency/throughput/memory, and outputs JSON results.

Usage:
    python tools/run_splade_baseline.py \\
        --index-dir data/beir_index_splade_scifact \\
        --queries data/beir/scifact/queries.jsonl \\
        --top-k 100 \\
        --num-samples 200 \\
        --output results/baselines/scifact/splade_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import psutil
import os


#!/usr/bin/env python
"""Run SPLADE queries on sparse vector index and measure performance.

Loads pre-built SPLADE index (sparse vectors) and executes batch queries,
measures latency/throughput/memory, and outputs JSON results.

Usage:
    python tools/run_splade_baseline.py \\
        --index-dir data/beir_index_splade_scifact \\
        --queries data/beir/scifact/queries.jsonl \\
        --top-k 100 \\
        --num-samples 200 \\
        --output results/baselines/scifact/splade_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

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


def run_splade_queries(index_dir: Path, queries: dict, top_k: int = 100) -> tuple:
    """Run queries against SPLADE sparse index."""
    
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        sys.exit(1)
    
    # Load index files
    index_file = index_dir / "splade_index.json"
    doc_ids_file = index_dir / "doc_ids.json"
    metadata_file = index_dir / "splade_metadata.json"
    
    if not all([index_file.exists(), doc_ids_file.exists(), metadata_file.exists()]):
        print(f"[ERROR] Missing index files in {index_dir}")
        return [], [], 0, 0
    
    print(f"Loading SPLADE index from {index_dir}...")
    with open(index_file) as f:
        sparse_vectors = json.load(f)
    
    with open(doc_ids_file) as f:
        doc_ids = json.load(f)
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"Index: {len(sparse_vectors)} sparse vectors")
    
    # Load SPLADE model for query encoding
    print(f"Loading SPLADE model for query encoding...")
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil", cache_dir=".cache")
    model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil", cache_dir=".cache")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Device: {device}")
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
        inputs = tokenizer(
            [query_text],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Compute sparse representation
        sparse_q = torch.relu(logits)
        sparse_q = torch.amax(sparse_q, dim=1).cpu().numpy()[0]  # [vocab_size]
        
        # Get non-zero indices and weights
        q_nonzero = {}
        for idx in np.nonzero(sparse_q)[0]:
            q_nonzero[int(idx)] = float(sparse_q[idx])
        
        # Score all documents
        scores = {}
        for doc_id in doc_ids:
            if doc_id in sparse_vectors:
                doc_sparse = sparse_vectors[doc_id]
                # Compute inner product (only over shared non-zero dimensions)
                score = 0.0
                for term_id_str, weight in doc_sparse.items():
                    term_id = int(term_id_str)
                    if term_id in q_nonzero:
                        score += q_nonzero[term_id] * weight
                scores[doc_id] = score
            else:
                scores[doc_id] = 0.0
        
        # Get top-k
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        doc_results = [(doc_id, score) for doc_id, score in top_docs]
        
        latency_ms = (time.time() - q_start) * 1000
        latencies.append(latency_ms)
        
        results_list.append((qid, doc_results, latency_ms))
        
        # Track memory
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = max(peak_memory, current_memory)
        
        if (i + 1) % max(10, len(queries) // 10) == 0:
            avg_lat = np.mean(latencies[-100:]) if len(latencies) >= 100 else np.mean(latencies)
            print(f"  [{i+1}/{len(queries)}] Avg latency: {avg_lat:.0f}ms, Peak memory: {peak_memory:.1f}GB")
    
    total_time = time.time() - start_time
    print(f"\n[OK] Completed {len(queries)} queries in {total_time:.1f}s")
    
    return results_list, latencies, initial_memory, peak_memory


def main():
    parser = argparse.ArgumentParser(description="Run SPLADE baseline queries")
    parser.add_argument("--index-dir", type=Path, required=True, help="Directory with SPLADE index")
    parser.add_argument("--queries", type=Path, required=True, help="Queries JSONL file")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k results")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit queries")
    parser.add_argument("--output", type=Path, default=Path("results/splade_baseline.json"))
    
    args = parser.parse_args()
    
    if not args.index_dir.exists():
        print(f"[ERROR] Index directory not found: {args.index_dir}")
        return 1
    
    if not args.queries.exists():
        print(f"[ERROR] Queries file not found: {args.queries}")
        return 1
    
    print("=" * 70)
    print("SPLADE BASELINE QUERIES")
    print("=" * 70)
    
    # Load queries
    queries = load_queries(args.queries, args.num_samples)
    print(f"[OK] Loaded {len(queries)} queries")
    
    # Run queries
    results_list, latencies, initial_memory, peak_memory = run_splade_queries(args.index_dir, queries, args.top_k)
    
    if not results_list:
        print("[ERROR] Query execution failed")
        return 1
    
    # Compile results JSON
    total_time = sum(lat for _, _, lat in results_list) / 1000  # Convert ms to s
    throughput = len(queries) / total_time if total_time > 0 else 0
    
    results = {
        "model": "splade-cocondenser-ensembledistil",
        "dataset": "unknown",
        "num_queries": len(queries),
        "performance": {
            "total_time_s": total_time,
            "throughput_qps": throughput,
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
    print(f"Throughput:      {throughput:.2f} QPS")
    print(f"Latency p50:     {np.median(latencies):.0f} ms")
    print(f"Latency p95:     {np.percentile(latencies, 95):.0f} ms")
    print(f"Peak Memory:     {peak_memory:.2f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



if __name__ == "__main__":
    main()
