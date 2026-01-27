#!/usr/bin/env python3
"""Generate latency vs corpus size sweep and plot results.

Usage: python tools/generate_latency_vs_size.py --dataset nfcorpus --sizes 1 2 4 --repeats 2

This script estimates docs-per-GB for the target corpus and uses `tools/system_metrics.py`
with `--limit` to index and measure query latencies for each sample size.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

# Ensure imports work when running script from subfolders or CI
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("paper/figs")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def estimate_docs_for_gb(corpus_path, target_gb):
    total_bytes = 0
    total_docs = 0
    with open(corpus_path, "rb") as f:
        for line in f:
            total_bytes += len(line)
            total_docs += 1
    if total_docs == 0:
        return 0
    bytes_per_doc = total_bytes / total_docs
    target_bytes = target_gb * 1024**3
    docs = int(min(total_docs, max(1, target_bytes // bytes_per_doc)))
    return docs


def run_system_metrics(corpus, index_dir, queries, top_k, limit):
    import sys

    script = Path(__file__).resolve().parents[1] / "tools" / "system_metrics.py"
    cmd = [
        sys.executable,
        str(script),
        "--corpus",
        str(corpus),
        "--index-dir",
        str(index_dir),
        "--queries",
        str(queries),
        "--top-k",
        str(top_k),
        "--limit",
        str(limit),
    ]
    print("Running:", " ".join(cmd))
    start = time.time()
    env = os.environ.copy()
    # Ensure repo root is visible to child processes
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run(cmd, check=True, env=env)
    duration = time.time() - start
    # Find latest system_metrics_*.json
    files = sorted(glob.glob("results/system_metrics_*.json"))
    assert files, "No system_metrics file generated"
    return files[-1], duration


def parse_system_metrics(fname):
    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    q = data.get("query", {})
    # latencies are in per_query (seconds)
    per_query = q.get("per_query", {})
    latencies_s = list(per_query.values())
    if not latencies_s:
        p50 = p95 = None
    else:
        lat_ms = [x * 1000.0 for x in latencies_s]
        p50 = statistics.median(lat_ms)
        p95 = max(lat_ms) if len(lat_ms) < 3 else statistics.quantiles(lat_ms, n=100)[94]
    peak_rss_mb = None
    if "peak_rss" in q:
        peak_rss_mb = q["peak_rss"] / (1024**2)
    return {"p50_ms": p50, "p95_ms": p95, "peak_rss_mb": peak_rss_mb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nfcorpus")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    corpus = base / f"data/beir/{args.dataset}/corpus.jsonl"
    queries = base / f"data/beir/{args.dataset}/queries_quick50.jsonl"
    if not queries.exists():
        queries = base / f"data/beir/{args.dataset}/queries.jsonl"
    if not corpus.exists() or not queries.exists():
        raise RuntimeError(
            f'Dataset files missing under {base / "data/beir"}; please download BEIR data or pick a dataset present in data/beir'
        )

    records = []
    for s in args.sizes:
        docs_limit = estimate_docs_for_gb(corpus, s)
        print(f"-- Target {s} GB => approx {docs_limit} docs")
        size_results = {"size_gb": s, "docs": docs_limit, "runs": []}
        for r in range(args.repeats):
            ts = int(time.time())
            index_dir = f"results/{args.dataset}_bench_index_{s}gb_run{r}_{ts}"
            out_file, dur = run_system_metrics(
                corpus, index_dir, queries, args.top_k, limit=docs_limit
            )
            parsed = parse_system_metrics(out_file)
            parsed["duration_s"] = dur
            p50_str = f"{parsed['p50_ms']:.1f}" if parsed["p50_ms"] is not None else "N/A"
            peak_str = (
                f"{parsed['peak_rss_mb']:.1f}" if parsed["peak_rss_mb"] is not None else "N/A"
            )
            print(f"  run {r+1}: p50={p50_str} ms, peak_rss={peak_str} MB")
            size_results["runs"].append(parsed)
            # small sleep to avoid racing files
            time.sleep(1)
        records.append(size_results)

    # Aggregate medians
    x = [r["size_gb"] for r in records]
    p50s = [
        (
            statistics.median([run["p50_ms"] for run in r["runs"] if run["p50_ms"] is not None])
            if any(run["p50_ms"] is not None for run in r["runs"])
            else None
        )
        for r in records
    ]
    p95s = [
        (
            statistics.median([run["p95_ms"] for run in r["runs"] if run["p95_ms"] is not None])
            if any(run["p95_ms"] is not None for run in r["runs"])
            else None
        )
        for r in records
    ]

    # Plot
    plt.figure(figsize=(6, 3))
    plt.plot(x, p50s, marker="o", label="p50 (ms)")
    plt.plot(x, p95s, marker="x", label="p95 (ms)")
    plt.xlabel("Corpus size (GB)")
    plt.ylabel("Query latency (ms)")
    plt.title("Query latency vs corpus size")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    out_png = PLOTS_DIR / f"latency_vs_corpus_size_{args.dataset}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved plot to", out_png)

    # Save CSV-like JSON summary
    summary = {"dataset": args.dataset, "records": records}
    with open(RESULTS_DIR / f"latency_vs_size_{args.dataset}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary JSON to results/")


if __name__ == "__main__":
    main()
