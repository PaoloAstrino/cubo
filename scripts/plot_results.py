#!/usr/bin/env python3
"""
Plot results from benchmark runner summary and per-run JSONs.

Generates the Part 2 graphs described in `how_to_compare.txt`:
- Win Rate per domain (success column as proxy)
- Latency vs dataset size
- Recall@K vs K
- Memory vs dataset size
- Ingestion time vs dataset size
- Compression ratio vs recall retention

Usage:
python scripts/plot_results.py --results-dir results/benchmark_runs --output-dir results/plots
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')


def load_summary_csv(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file {summary_path} not found")
    return pd.read_csv(summary_path)


def load_run_jsons(results_dir: Path):
    runs = []
    for run_dir in results_dir.iterdir():
        if run_dir.is_dir():
            run_file = run_dir / 'benchmark_run.json'
            if run_file.exists():
                try:
                    with open(run_file, 'r', encoding='utf-8') as f:
                        runs.append(json.load(f))
                except Exception as e:
                    print(f"Failed to parse {run_file}: {e}")
    return runs


def extract_dataset_size(run_json):
    # Try various paths for the data size
    try:
        ingestion = run_json['metadata'].get('ingestion_results')
        if ingestion and 'ingestion' in ingestion:
            return ingestion['ingestion']['data_size']['total_gb']
    except Exception:
        pass
    # Attempt to find any total_gb under 'results'
    try:
        # scan metadata for any key with 'total_gb'
        md = run_json.get('results', {}).get('metadata', {})
        for k, v in md.items():
            if isinstance(v, dict) and 'total_gb' in v:
                return v['total_gb']
    except Exception:
        pass
    return None


def plot_win_rate_by_domain(df, outdir: Path):
    plt.figure(figsize=(8, 5))
    # Aggregate by dataset and config
    agg = df.groupby(['dataset', 'retrieval_config']).agg({'success': 'mean'}).reset_index()
    sns.barplot(data=agg, x='dataset', y='success', hue='retrieval_config')
    plt.title('Win Rate (Success proxy) by Domain and Config')
    plt.ylim(0, 1)
    plt.ylabel('Win Rate (0-1)')
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / 'win_rate_by_domain.png', bbox_inches='tight')
    plt.close()


def plot_latency_vs_size(df, runs, outdir: Path):
    plt.figure(figsize=(8, 5))
    sizes = []
    latencies = []
    labels = []
    for r in runs:
        size = extract_dataset_size(r)
        latency = r.get('results', {}).get('metadata', {}).get('avg_retrieval_latency_p50_ms', None)
        if size is not None and latency is not None:
            sizes.append(size)
            latencies.append(latency)
            labels.append(r['metadata'].get('retrieval_config', {}).get('name', 'config'))
    if not sizes:
        print("No size/latency data available for plot_latency_vs_size")
        return
    sns.scatterplot(x=sizes, y=latencies, hue=labels)
    plt.xlabel('Dataset Size (GB)')
    plt.ylabel('Avg retrieval latency p50 (ms)')
    plt.title('Retrieval latency vs dataset size')
    plt.savefig(outdir / 'latency_vs_dataset_size.png', bbox_inches='tight')
    plt.close()


def plot_recall_vs_k(df, runs, outdir: Path):
    # Find keys like 'avg_recall_at_k_X' in summary CSV or run metadata
    recall_cols = [c for c in df.columns if c.startswith('avg_recall_at_k_')]
    if not recall_cols:
        # Fallback: scan run JSONs for recall values
        all_pts = []
        for r in runs:
            md = r.get('results', {}).get('metadata', {})
            for k, v in md.items():
                if k.startswith('avg_recall_at_k_'):
                    kval = int(k.split('_')[-1])
                    all_pts.append({'k': kval, 'recall': v, 'dataset': r['metadata']['dataset'].get('name', '')})
        if not all_pts:
            print("No Recall@K data found for plotting")
            return
        rdf = pd.DataFrame(all_pts)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=rdf, x='k', y='recall', hue='dataset', marker='o')
        plt.title('Recall@K across datasets')
        plt.xlabel('K')
        plt.ylabel('Recall@K')
        plt.savefig(outdir / 'recall_vs_k.png', bbox_inches='tight')
        plt.close()
        return

    # If summary CSV has recall columns, plot them aggregated by dataset
    k_values = sorted([int(c.split('_')[-1]) for c in recall_cols])
    plt.figure(figsize=(8, 5))
    for ds in df['dataset'].unique():
        ds_df = df[df['dataset'] == ds]
        # average recall per k
        means = []
        for k in k_values:
            col = f'avg_recall_at_k_{k}'
            if col in ds_df.columns:
                means.append(ds_df[col].mean())
            else:
                means.append(float('nan'))
        sns.lineplot(x=k_values, y=means, label=ds, marker='o')
    plt.title('Recall@K')
    plt.xlabel('K')
    plt.ylabel('Recall@K (avg)')
    plt.legend()
    plt.savefig(outdir / 'recall_vs_k_summary.png', bbox_inches='tight')
    plt.close()


def plot_memory_vs_size(df, runs, outdir: Path):
    plt.figure(figsize=(8, 5))
    sizes = []
    memory = []
    for r in runs:
        size = extract_dataset_size(r)
        try:
            mem_delta = r['metadata']['ingestion_results']['ingestion'].get('memory_delta_gb')
        except Exception:
            mem_delta = None
        if size is not None and mem_delta is not None:
            sizes.append(size)
            memory.append(mem_delta)
    if not sizes:
        print('No memory vs size data available')
        return
    sns.scatterplot(x=sizes, y=memory)
    plt.title('Memory delta (GB) vs dataset size (GB)')
    plt.xlabel('Dataset Size (GB)')
    plt.ylabel('Memory Delta (GB)')
    plt.savefig(outdir / 'memory_vs_size.png', bbox_inches='tight')
    plt.close()


def plot_ingestion_time(df, runs, outdir: Path):
    plt.figure(figsize=(8, 5))
    sizes = []
    mins_per_gb = []
    for r in runs:
        size = extract_dataset_size(r)
        try:
            mpergb = r['metadata']['ingestion_results']['ingestion'].get('minutes_per_gb')
        except Exception:
            mpergb = None
        if size is not None and mpergb is not None:
            sizes.append(size)
            mins_per_gb.append(mpergb)
    if not sizes:
        print('No ingestion time data')
        return
    sns.lineplot(x=sizes, y=mins_per_gb, marker='o')
    plt.title('Ingestion time (min per GB) vs dataset size')
    plt.xlabel('Dataset Size (GB)')
    plt.ylabel('Min / GB')
    plt.savefig(outdir / 'ingestion_time_vs_size.png', bbox_inches='tight')
    plt.close()


def plot_compression_ratio_vs_recall(df, runs, outdir: Path):
    plt.figure(figsize=(8, 5))
    ratios = []
    recalls = []
    for r in runs:
        try:
            compression = r['metadata']['ingestion_results']['ingestion'].get('compression_ratio')
            recall = r['results']['metadata'].get('avg_recall_at_k_10')
        except Exception:
            compression = None
            recall = None
        if compression is not None and recall is not None:
            ratios.append(compression)
            recalls.append(recall)
    if not ratios:
        print('No compression/recall data')
        return
    sns.scatterplot(x=ratios, y=recalls)
    plt.title('Compression ratio vs Recall@10')
    plt.xlabel('Compression Ratio')
    plt.ylabel('Recall@10')
    plt.savefig(outdir / 'compression_vs_recall.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--results-dir', default='results/benchmark_runs', help='Directory with benchmark_run.json and summary.csv')
    parser.add_argument('--output-dir', default='results/plots', help='Output directory for plots')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / 'summary.csv'
    try:
        df = load_summary_csv(summary_path)
    except Exception as e:
        print(f"Failed to load summary.csv: {e}")
        return

    runs = load_run_jsons(results_dir)

    plot_win_rate_by_domain(df, outdir)
    plot_latency_vs_size(df, runs, outdir)
    plot_recall_vs_k(df, runs, outdir)
    plot_memory_vs_size(df, runs, outdir)
    plot_ingestion_time(df, runs, outdir)
    plot_compression_ratio_vs_recall(df, runs, outdir)

    print('Plots saved to', outdir)


if __name__ == '__main__':
    main()
