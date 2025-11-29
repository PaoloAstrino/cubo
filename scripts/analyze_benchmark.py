#!/usr/bin/env python3
"""
Analyze BEIR benchmark results and produce comprehensive metrics report.

Usage:
    python scripts/analyze_benchmark.py results/beir_200_benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def compute_ir_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute aggregate IR metrics from individual results."""
    metrics = {}
    k_values = [5, 10, 20, 50, 100]
    
    for k in k_values:
        recalls = []
        ndcgs = []
        mrrs = []
        
        for r in results:
            ir = r.get('ir_metrics', {})
            if f'recall_at_k_{k}' in ir:
                recalls.append(ir[f'recall_at_k_{k}'])
            if f'ndcg_at_k_{k}' in ir:
                ndcgs.append(ir[f'ndcg_at_k_{k}'])
            if f'mrr_at_k_{k}' in ir:
                mrrs.append(ir[f'mrr_at_k_{k}'])
        
        if recalls:
            metrics[f'recall@{k}'] = sum(recalls) / len(recalls)
            metrics[f'recall@{k}_hits'] = len([r for r in recalls if r > 0])
        if ndcgs:
            metrics[f'ndcg@{k}'] = sum(ndcgs) / len(ndcgs)
        if mrrs:
            metrics[f'mrr@{k}'] = sum(mrrs) / len(mrrs)
    
    return metrics


def compute_latency_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute latency statistics."""
    latencies = []
    for r in results:
        lat = r.get('retrieval_latency', {})
        if 'p50_ms' in lat:
            latencies.append(lat['p50_ms'])
    
    if not latencies:
        return {}
    
    return {
        'latency_p50_ms': statistics.median(latencies),
        'latency_mean_ms': statistics.mean(latencies),
        'latency_min_ms': min(latencies),
        'latency_max_ms': max(latencies),
        'latency_stdev_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
    }


def analyze_benchmark(filepath: str) -> Dict[str, Any]:
    """Analyze benchmark results and return comprehensive metrics."""
    data = load_results(filepath)
    
    all_results = []
    for difficulty in ['easy', 'medium', 'hard']:
        all_results.extend(data.get('results', {}).get(difficulty, []))
    
    analysis = {
        'summary': {
            'total_questions': len(all_results),
            'successful': sum(1 for r in all_results if r.get('success')),
            'failed': sum(1 for r in all_results if not r.get('success')),
            'success_rate': sum(1 for r in all_results if r.get('success')) / len(all_results) if all_results else 0,
        },
        'ir_metrics': compute_ir_metrics(all_results),
        'latency_metrics': compute_latency_metrics(all_results),
        'hardware': data.get('metadata', {}).get('hardware', {}),
    }
    
    return analysis


def print_report(analysis: Dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print("=" * 70)
    print("CUBO RAG BENCHMARK ANALYSIS REPORT")
    print("=" * 70)
    
    # Summary
    print("\n📊 SUMMARY")
    print("-" * 40)
    s = analysis['summary']
    print(f"  Total Questions:  {s['total_questions']}")
    print(f"  Successful:       {s['successful']} ({s['success_rate']*100:.1f}%)")
    print(f"  Failed:           {s['failed']}")
    
    # IR Metrics
    print("\n📈 INFORMATION RETRIEVAL METRICS")
    print("-" * 40)
    ir = analysis['ir_metrics']
    
    print("\n  Recall@K (higher is better):")
    for k in [5, 10, 20, 50, 100]:
        if f'recall@{k}' in ir:
            hits = ir.get(f'recall@{k}_hits', 0)
            print(f"    Recall@{k:3d}: {ir[f'recall@{k}']:.4f}  (hits: {hits})")
    
    print("\n  nDCG@K (higher is better):")
    for k in [5, 10, 20, 50, 100]:
        if f'ndcg@{k}' in ir:
            print(f"    nDCG@{k:3d}:   {ir[f'ndcg@{k}']:.4f}")
    
    # Latency Metrics
    print("\n⏱️  LATENCY METRICS")
    print("-" * 40)
    lat = analysis['latency_metrics']
    if lat:
        print(f"  Median (p50):    {lat.get('latency_p50_ms', 0):.1f} ms")
        print(f"  Mean:            {lat.get('latency_mean_ms', 0):.1f} ms")
        print(f"  Min:             {lat.get('latency_min_ms', 0):.1f} ms")
        print(f"  Max:             {lat.get('latency_max_ms', 0):.1f} ms")
        print(f"  Std Dev:         {lat.get('latency_stdev_ms', 0):.1f} ms")
    
    # Hardware
    print("\n🖥️  HARDWARE CONFIGURATION")
    print("-" * 40)
    hw = analysis['hardware']
    if hw:
        cpu = hw.get('cpu', {})
        ram = hw.get('ram', {})
        gpu = hw.get('gpu', {})
        print(f"  CPU:    {cpu.get('model', 'Unknown')}")
        print(f"  Cores:  {cpu.get('cores_physical', '?')} physical / {cpu.get('cores_logical', '?')} logical")
        print(f"  RAM:    {ram.get('total_gb', 0):.1f} GB")
        if gpu.get('available'):
            print(f"  GPU:    {gpu.get('device_name', 'Unknown')}")
            print(f"  VRAM:   {gpu.get('vram_total_gb', 0):.1f} GB")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze BEIR benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of formatted text")
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)
    
    analysis = analyze_benchmark(args.results_file)
    
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_report(analysis)


if __name__ == "__main__":
    main()
