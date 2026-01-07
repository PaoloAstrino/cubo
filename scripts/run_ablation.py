#!/usr/bin/env python3
"""Run ablation suite: dense-only, bm25-only, hybrid for specified datasets."""
import argparse
import subprocess
from pathlib import Path

MODES = {
    "dense": ["--use-optimized", "--laptop-mode"],
    "hybrid": [],
    "bm25": ["--bm25-only"],
}


def run_mode(dataset, dataset_corpus, dataset_queries, mode, top_k, index_dir, output_base, force_reindex=False):
    out = Path(output_base) / f"beir_run_{dataset}_topk{top_k}_{mode}.json"
    
    cmd = [
        "python",
        "scripts/run_beir_adapter.py",
        "--corpus",
        str(dataset_corpus),
        "--queries",
        str(dataset_queries),
        "--output",
        str(out),
        "--index-dir",
        str(index_dir),
        "--top-k",
        str(top_k),
    ]
    
    # Only reindex if it's the first run or force_reindex is True
    # We'll handle this in the main loop
    if force_reindex:
        cmd.append("--reindex")

    # Append mode-specific flags
    cmd += MODES[mode]

    print(f"Running mode {mode}:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation modes for BEIR datasets")
    parser.add_argument("--dataset", required=True, help="dataset name (e.g., nfcorpus, fiqa)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--index-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--queries", help="Override queries path")
    parser.add_argument("--corpus", help="Override corpus path")
    parser.add_argument("--force-reindex", action="store_true", help="Force reindexing")

    args = parser.parse_args()
    dataset = args.dataset

    corpus = Path(args.corpus) if args.corpus else Path(f"data/beir/{dataset}/corpus.jsonl")
    queries = Path(args.queries) if args.queries else Path(f"data/beir/{dataset}/queries.jsonl")

    # Single shared index for this dataset/top_k combo
    shared_index_dir = Path(args.index_dir) / f"beir_index_{dataset}_shared"
    
    # We only need to index once for the entire ablation suite
    first_run = True
    for mode in ["dense", "bm25", "hybrid"]:
        # Reindex on first mode run if directory doesn't exist or force_reindex is set
        do_reindex = (first_run and (not shared_index_dir.exists() or args.force_reindex))
        
        run_mode(dataset, corpus, queries, mode, args.top_k, shared_index_dir, args.output_dir, force_reindex=do_reindex)
        first_run = False

    print("Ablation runs completed for", dataset)
