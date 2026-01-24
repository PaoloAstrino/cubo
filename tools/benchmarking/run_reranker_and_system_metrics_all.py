#!/usr/bin/env python3
"""Run reranker eval and system metrics for selected datasets, save per-dataset summaries.

Usage: python tools/run_reranker_and_system_metrics_all.py
"""
import subprocess
import glob
import os
import shutil
import time

datasets = [
    'fiqa', 'scifact', 'arguana', 'nfcorpus', 'ultradomain_medium', 'ultradomain_legal', 'ultradomain_politics', 'ultradomain_agriculture', 'scidocs', 'ragbench_merged'
]

def find_index_dir(ds):
    # common patterns
    candidates = [f'results/beir_index_{ds}', f'results/beir_index_{ds}_shared', f'results/beir_index_{ds}_topk50', f'results/beir_index_{ds}_shared']
    for c in candidates:
        if os.path.exists(c):
            return c
    # try any folder under results that contains ds
    for d in glob.glob('results/*'):
        if ds in d and os.path.isdir(d) and 'index' in d:
            return d
    # fallback to top-level beir_index_*
    for d in glob.glob('results/beir_index*'):
        if ds in d:
            return d
    return None


def find_queries(ds):
    candidates = [f'data/beir/{ds}/queries.jsonl', f'data/beir/{ds}/queries_test100.jsonl', f'data/beir/{ds}/queries_quick50.jsonl']
    for c in candidates:
        if os.path.exists(c):
            return c
    # ultradomain variants
    for prefix in ['ultradomain_medium','ultradomain_legal','ultradomain_politics','ultradomain_agriculture']:
        if ds.startswith('ultradomain'):
            p=f'data/beir/{ds}/queries.jsonl'
            if os.path.exists(p):
                return p
    return None


def run_reranker(ds, index_dir, queries, top_k=10):
    print('Running reranker eval for', ds)
    cmd = ['python','tools/run_reranker_eval.py','--index-dir', index_dir, '--queries', queries, '--top-k', str(top_k)]
    subprocess.run(cmd, check=True)
    # move outputs
    summary = f'results/reranker_eval_topk{top_k}.json'
    if os.path.exists(summary):
        dst = f'results/reranker_eval_{ds}_topk{top_k}.json'
        shutil.move(summary, dst)
    for prefix in [f'results/run_no_rerank_topk{top_k}.json', f'results/run_with_rerank_topk{top_k}.json', f'results/run_no_rerank_topk{top_k}_latencies.json', f'results/run_with_rerank_topk{top_k}_latencies.json']:
        if os.path.exists(prefix):
            dst = prefix.replace('run_', f'run_{ds}_')
            shutil.move(prefix, dst)
    print('Saved reranker results to', dst)


def run_system_metrics(ds, corpus, index_dir, queries, top_k=50, limit=0):
    print('Running system metrics for', ds)
    cmd = ['python','tools/system_metrics.py','--corpus', corpus, '--index-dir', index_dir, '--queries', queries, '--top-k', str(top_k)]
    if limit:
        cmd += ['--limit', str(limit)]
    subprocess.run(cmd, check=True)
    # find latest file
    candidates = sorted(glob.glob('results/system_metrics_*.json'))
    if candidates:
        src = candidates[-1]
        ts = int(time.time())
        dst = f'results/system_metrics_{ds}_{ts}.json'
        shutil.move(src, dst)
        print('Saved system metrics to', dst)
    else:
        print('No system metrics file found')


if __name__ == '__main__':
    results = []
    for ds in datasets:
        index_dir = find_index_dir(ds)
        queries = find_queries(ds)
        # corpus
        corpus = f'data/beir/{ds}/corpus.jsonl'
        if not os.path.exists(corpus):
            # try alternate subdirs
            if ds.startswith('ultradomain'):
                # map
                corpus = f'data/beir/{ds}/corpus.jsonl'
        if not index_dir or not queries:
            print('Skipping', ds, 'missing index_dir or queries (index_dir:', index_dir, 'queries:', queries, ')')
            continue
        try:
            run_reranker(ds, index_dir, queries, top_k=10)
        except Exception as e:
            print('Reranker failed for', ds, e)
        try:
            run_system_metrics(ds, corpus, index_dir, queries, top_k=10)
        except Exception as e:
            print('System metrics failed for', ds, e)
    print('All done')
