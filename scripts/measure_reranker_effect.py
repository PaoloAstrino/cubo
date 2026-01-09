#!/usr/bin/env python3
"""
Simple reranker and system metrics collector.
Uses existing run_beir_adapter.py with/without reranking.
"""
import argparse
import json
import subprocess
import time
import os
from pathlib import Path

def measure_reranker_effect(dataset, index_dir, queries, qrels, top_k=50):
    """
    Run retrieval with and without reranker, measure ΔRecall and resource usage.
    """
    print(f"\n=== Measuring reranker effect for {dataset} ===")
    
    # Run base (dense-only, no rerank)
    out_base = f"results/rerank_study_{dataset}_base.json"
    cmd_base = [
        'python', 'scripts/run_beir_adapter.py',
        '--corpus', f'data/beir/{dataset}/corpus.jsonl',
        '--queries', queries,
        '--output', out_base,
        '--index-dir', index_dir,
        '--top-k', str(top_k),
        '--use-optimized', '--laptop-mode',  # no reranker
    ]
    
    print(f"Running base (no rerank): {' '.join(cmd_base)}")
    start = time.time()
    result_base = subprocess.run(cmd_base, capture_output=True, text=True)
    elapsed_base = time.time() - start
    
    if result_base.returncode != 0:
        print(f"ERROR: base run failed for {dataset}")
        print(result_base.stderr)
        return None
    
    # Compute metrics for base
    metrics_base_file = out_base.replace('.json', '_metrics_k10.json')
    subprocess.run([
        'python', 'scripts/calculate_beir_metrics.py',
        '--results', out_base,
        '--qrels', qrels,
        '--k', '10'
    ], check=True)
    
    with open(metrics_base_file, 'r', encoding='utf-8') as f:
        metrics_base = json.load(f)
    
    # Run with reranker (remove --use-optimized and --laptop-mode)
    out_rerank = f"results/rerank_study_{dataset}_reranked.json"
    cmd_rerank = [
        'python', 'scripts/run_beir_adapter.py',
        '--corpus', f'data/beir/{dataset}/corpus.jsonl',
        '--queries', queries,
        '--output', out_rerank,
        '--index-dir', index_dir,
        '--top-k', str(top_k),
        # no --use-optimized, no --laptop-mode -> enables reranker
    ]
    
    print(f"Running with reranker: {' '.join(cmd_rerank)}")
    start = time.time()
    result_rerank = subprocess.run(cmd_rerank, capture_output=True, text=True)
    elapsed_rerank = time.time() - start
    
    if result_rerank.returncode != 0:
        print(f"WARNING: rerank run failed for {dataset}, skipping reranker measurement")
        print(result_rerank.stderr)
        # Return base-only result
        return {
            'dataset': dataset,
            'base_recall': metrics_base['recall_at_k'],
            'base_mrr': metrics_base['mrr'],
            'base_ndcg': metrics_base['ndcg'],
            'base_time_s': elapsed_base,
            'rerank_failed': True,
            'note': 'Reranker run failed or unavailable'
        }
    
    # Compute metrics for rerank
    metrics_rerank_file = out_rerank.replace('.json', '_metrics_k10.json')
    subprocess.run([
        'python', 'scripts/calculate_beir_metrics.py',
        '--results', out_rerank,
        '--qrels', qrels,
        '--k', '10'
    ], check=True)
    
    with open(metrics_rerank_file, 'r', encoding='utf-8') as f:
        metrics_rerank = json.load(f)
    
    # Compute delta
    delta_recall = metrics_rerank['recall_at_k'] - metrics_base['recall_at_k']
    delta_mrr = metrics_rerank['mrr'] - metrics_base['mrr']
    
    result = {
        'dataset': dataset,
        'base_recall': metrics_base['recall_at_k'],
        'base_mrr': metrics_base['mrr'],
        'base_ndcg': metrics_base['ndcg'],
        'base_time_s': elapsed_base,
        'rerank_recall': metrics_rerank['recall_at_k'],
        'rerank_mrr': metrics_rerank['mrr'],
        'rerank_ndcg': metrics_rerank['ndcg'],
        'rerank_time_s': elapsed_rerank,
        'delta_recall': delta_recall,
        'delta_mrr': delta_mrr,
        'time_overhead_s': elapsed_rerank - elapsed_base,
        'note': 'Peak RAM measurement requires psutil monitoring; estimate >16GB for reranker'
    }
    
    print(f"\n✅ {dataset} results:")
    print(f"  Base Recall@10: {metrics_base['recall_at_k']:.4f} ({elapsed_base:.1f}s)")
    print(f"  Rerank Recall@10: {metrics_rerank['recall_at_k']:.4f} ({elapsed_rerank:.1f}s)")
    print(f"  ΔRecall: {delta_recall:+.4f}, Time overhead: {elapsed_rerank - elapsed_base:.1f}s")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure reranker effect')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., nfcorpus, fiqa)')
    parser.add_argument('--index-dir', help='Index directory (default: results/beir_index_{dataset}_shared)')
    parser.add_argument('--queries', help='Queries file (default: data/beir/{dataset}/queries.jsonl)')
    parser.add_argument('--qrels', help='Qrels file (default: data/beir/{dataset}/qrels/test.tsv)')
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()
    
    dataset = args.dataset
    index_dir = args.index_dir or f'results/beir_index_{dataset}_shared'
    queries = args.queries or f'data/beir/{dataset}/queries.jsonl'
    qrels = args.qrels or f'data/beir/{dataset}/qrels/test.tsv'
    
    result = measure_reranker_effect(dataset, index_dir, queries, qrels, args.top_k)
    
    if result:
        # Save to results/reranker_summary_{dataset}.json
        out_file = f'results/reranker_summary_{dataset}.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved reranker summary to {out_file}")
