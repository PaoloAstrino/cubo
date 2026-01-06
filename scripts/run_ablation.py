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


def run_mode(dataset_corpus, dataset_queries, mode, top_k, index_dir_base, output_base):
    out = Path(output_base) / f"beir_run_{dataset}_topk{top_k}_{mode}.json"
    idx = Path(index_dir_base) / f"beir_index_{dataset}_{mode}_topk{top_k}"

    cmd = [
        "python",
        "scripts/run_beir_adapter.py",
        "--corpus",
        str(dataset_corpus),
        "--queries",
        str(dataset_queries),
        "--reindex",
        "--output",
        str(out),
        "--index-dir",
        str(idx),
        "--top-k",
        str(top_k),
    ]

    # Append mode-specific flags
    cmd += MODES[mode]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation modes for BEIR datasets")
    parser.add_argument("--dataset", required=True, help="dataset name (e.g., nfcorpus, fiqa)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--index-dir", default="results")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()
    dataset = args.dataset

    corpus = Path(f"data/beir/{dataset}/corpus.jsonl")
    queries = Path(f"data/beir/{dataset}/queries.jsonl")

    for mode in ["dense", "bm25", "hybrid"]:
        run_mode(corpus, queries, mode, args.top_k, args.index_dir, args.output_dir)

    print("Ablation runs completed for", dataset)
