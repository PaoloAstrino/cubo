#!/usr/bin/env python3
"""
Simple reranker and system metrics collector.
Uses existing run_beir_adapter.py with/without reranking.
"""
import argparse
import json
import subprocess
import time


def _build_base_command(dataset, queries, index_dir, top_k, output_file):
    """Build command for base retrieval without reranker."""
    return [
        'python', 'tools/run_beir_adapter.py',
        '--corpus', f'data/beir/{dataset}/corpus.jsonl',
        '--queries', queries,
        '--output', output_file,
        '--index-dir', index_dir,
        '--top-k', str(top_k),
        '--use-optimized', '--laptop-mode',
    ]


def _build_rerank_command(dataset, queries, index_dir, top_k, output_file):
    """Build command for retrieval with reranker."""
    return [
        'python', 'tools/run_beir_adapter.py',
        '--corpus', f'data/beir/{dataset}/corpus.jsonl',
        '--queries', queries,
        '--output', output_file,
        '--index-dir', index_dir,
        '--top-k', str(top_k),
    ]


def _run_retrieval(cmd, description):
    """Run retrieval command and return timing and result."""
    print(f"Running {description}: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    return result, elapsed


def _compute_metrics(results_file, qrels, k=10):
    """Compute metrics for results file."""
    metrics_file = results_file.replace('.json', f'_metrics_k{k}.json')
    subprocess.run([
        'python', 'tools/calculate_beir_metrics.py',
        '--results', results_file,
        '--qrels', qrels,
        '--k', str(k)
    ], check=True)
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _create_base_only_result(dataset, metrics_base, elapsed_base):
    """Create result dictionary for base-only (when reranker fails)."""
    return {
        'dataset': dataset,
        'base_recall': metrics_base['recall_at_k'],
        'base_mrr': metrics_base['mrr'],
        'base_ndcg': metrics_base['ndcg'],
        'base_time_s': elapsed_base,
        'rerank_failed': True,
        'note': 'Reranker run failed or unavailable'
    }


def _create_full_result(dataset, metrics_base, elapsed_base, metrics_rerank, elapsed_rerank):
    """Create full result dictionary comparing base and reranker."""
    delta_recall = metrics_rerank['recall_at_k'] - metrics_base['recall_at_k']
    delta_mrr = metrics_rerank['mrr'] - metrics_base['mrr']
    
    return {
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


def _print_results(dataset, metrics_base, elapsed_base, metrics_rerank, elapsed_rerank, delta_recall):
    """Print formatted results."""
    print(f"\n✅ {dataset} results:")
    print(f"  Base Recall@10: {metrics_base['recall_at_k']:.4f} ({elapsed_base:.1f}s)")
    print(f"  Rerank Recall@10: {metrics_rerank['recall_at_k']:.4f} ({elapsed_rerank:.1f}s)")
    print(f"  ΔRecall: {delta_recall:+.4f}, Time overhead: {elapsed_rerank - elapsed_base:.1f}s")


def measure_reranker_effect(dataset, index_dir, queries, qrels, top_k=50):
    """Run retrieval with and without reranker, measure ΔRecall and resource usage."""
    print(f"\n=== Measuring reranker effect for {dataset} ===")
    
    # Run base (no reranker)
    out_base = f"results/rerank_study_{dataset}_base.json"
    cmd_base = _build_base_command(dataset, queries, index_dir, top_k, out_base)
    result_base, elapsed_base = _run_retrieval(cmd_base, "base (no rerank)")
    
    if result_base.returncode != 0:
        print(f"ERROR: base run failed for {dataset}")
        print(result_base.stderr)
        return None
    
    metrics_base = _compute_metrics(out_base, qrels, k=10)
    
    # Run with reranker
    out_rerank = f"results/rerank_study_{dataset}_reranked.json"
    cmd_rerank = _build_rerank_command(dataset, queries, index_dir, top_k, out_rerank)
    result_rerank, elapsed_rerank = _run_retrieval(cmd_rerank, "with reranker")
    
    if result_rerank.returncode != 0:
        print(f"WARNING: rerank run failed for {dataset}, skipping reranker measurement")
        print(result_rerank.stderr)
        return _create_base_only_result(dataset, metrics_base, elapsed_base)
    
    metrics_rerank = _compute_metrics(out_rerank, qrels, k=10)
    
    result = _create_full_result(dataset, metrics_base, elapsed_base, metrics_rerank, elapsed_rerank)
    _print_results(dataset, metrics_base, elapsed_base, metrics_rerank, elapsed_rerank, result['delta_recall'])
    
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
