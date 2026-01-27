#!/usr/bin/env python3
"""Collect system metrics: indexing peak RSS/time and query p50/p95/peak RSS."""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil


def monitor_process_and_wait(proc):
    p = psutil.Process(proc.pid)
    peak = 0
    while True:
        if proc.poll() is not None:
            break
        try:
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    try:
        rss = p.memory_info().rss
        if rss > peak:
            peak = rss
    except Exception:
        pass
    return peak


def run_index_worker(corpus, index_dir, limit=0):
    # Use the same Python executable and module invocation to preserve package imports
    cmd = [sys.executable, "-m", "tools.worker_index", "--corpus", corpus, "--index-dir", index_dir]
    if limit:
        cmd += ["--limit", str(limit)]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    peak = monitor_process_and_wait(proc)
    if proc.returncode != 0:
        raise RuntimeError("index worker failed")
    # Read metrics file
    with open(index_dir.rstrip("/") + "/index_metrics.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    m["peak_rss"] = peak
    return m


def run_query_benchmark(index_dir, queries, top_k):
    # Reuse worker_retrieve to get latencies and run
    out = f"results/system_query_run_topk{top_k}.json"
    cmd = [
        sys.executable,
        "-m",
        "tools.worker_retrieve",
        "--index-dir",
        index_dir,
        "--queries",
        queries,
        "--output",
        out,
        "--top-k",
        str(top_k),
        "--mode",
        "no_rerank",
    ]
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    peak = monitor_process_and_wait(proc)
    if proc.returncode != 0:
        raise RuntimeError("query worker failed")
    lat_file = out.replace(".json", "_latencies.json")
    with open(lat_file, "r", encoding="utf-8") as f:
        lat = json.load(f)
    lat["peak_rss"] = peak
    return lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus")
    parser.add_argument("--index-dir")
    parser.add_argument("--queries")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    index_metrics = run_index_worker(args.corpus, args.index_dir, limit=args.limit)
    query_metrics = run_query_benchmark(args.index_dir, args.queries, args.top_k)

    out = {"indexing": index_metrics, "query": query_metrics}
    with open(f"results/system_metrics_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved system metrics")
