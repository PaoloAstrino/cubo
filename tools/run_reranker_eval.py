#!/usr/bin/env python3
"""Run retrieval with and without reranker, measure recall delta and resource usage."""
import argparse
import json
import subprocess
import time
import os
from statistics import median
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
    # final check
    try:
        rss = p.memory_info().rss
        if rss > peak:
            peak = rss
    except Exception:
        pass
    return peak


def compute_latency_stats(lat_file):
    with open(lat_file, 'r', encoding='utf-8') as f:
        d = json.load(f)
    times = list(d.get('per_query', {}).values())
    times = [t for t in times if t is not None]
    if not times:
        return {'p50': 0, 'p95': 0, 'total': d.get('total_time_s', 0)}
    times_sorted = sorted(times)
    p50 = times_sorted[int(0.5 * len(times_sorted))]
    p95 = times_sorted[int(0.95 * len(times_sorted))] if len(times_sorted) > 1 else times_sorted[-1]
    return {'p50': p50, 'p95': p95, 'total': d.get('total_time_s', 0)}


def run_worker(index_dir, queries, output, top_k, mode):
    cmd = [
        'python', 'tools/worker_retrieve.py',
        '--index-dir', index_dir,
        '--queries', queries,
        '--output', output,
        '--top-k', str(top_k),
        '--mode', mode,
    ]
    proc = subprocess.Popen(cmd)
    peak = monitor_process_and_wait(proc)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed with code {proc.returncode}")
    return peak


def run_eval(index_dir, queries, top_k):
    out_no = f"results/run_no_rerank_topk{top_k}.json"
    out_with = f"results/run_with_rerank_topk{top_k}.json"

    peak_no = run_worker(index_dir, queries, out_no, top_k, 'no_rerank')
    lat_no = compute_latency_stats(out_no.replace('.json', '_latencies.json'))

    peak_with = run_worker(index_dir, queries, out_with, top_k, 'with_rerank')
    lat_with = compute_latency_stats(out_with.replace('.json', '_latencies.json'))

    # Compute metrics
    # Use tools/calc_metrics_from_run.py to compute Recall@10
    res_no = subprocess.run(['python', 'tools/calc_metrics_from_run.py', '--run', out_no, '--qrels', queries.replace('queries.jsonl','qrels/test.tsv'), '--k', '10'], capture_output=True, text=True)
    res_with = subprocess.run(['python', 'tools/calc_metrics_from_run.py', '--run', out_with, '--qrels', queries.replace('queries.jsonl','qrels/test.tsv'), '--k', '10'], capture_output=True, text=True)

    # Parse printed outputs (the script prints Recall/MRR lines)
    print('--- No Rerank ---')
    print(res_no.stdout)
    print('--- With Rerank ---')
    print(res_with.stdout)

    summary = {
        'no_rerank': {'peak_rss': peak_no, 'latency': lat_no},
        'with_rerank': {'peak_rss': peak_with, 'latency': lat_with},
    }
    with open(f'results/reranker_eval_topk{top_k}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Saved reranker eval summary to', f'results/reranker_eval_topk{top_k}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()
    run_eval(args.index_dir, args.queries, args.top_k)
