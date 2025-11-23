import os
import csv
import json
from pathlib import Path
from scripts.plot_results import main as plot_main
import subprocess


def create_fake_results(tmp_path: Path):
    results_dir = tmp_path / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create summary.csv
    summary_csv = results_dir / 'summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_id','dataset','retrieval_config','ablation','mode','timestamp','success','avg_recall_at_k_10','avg_ndcg_at_k_10','avg_retrieval_latency_p50_ms','avg_answer_relevance','ingestion_minutes_per_gb'])
        writer.writeheader()
        writer.writerow({'run_id':'run1','dataset':'smoke','retrieval_config':'hybrid_test','ablation':'none','mode':'retrieval-only','timestamp':123,'success':1.0,'avg_recall_at_k_10':0.9,'avg_ndcg_at_k_10':0.88,'avg_retrieval_latency_p50_ms':60,'avg_answer_relevance':0.9,'ingestion_minutes_per_gb':1.5})

    # Create a run dir with benchmark_run.json
    run_dir = results_dir / 'run1'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = {
        'metadata': { 'run_id': 'run1', 'dataset': {'path':'data/smoke','name':'smoke'}, 'retrieval_config': {'name':'hybrid_test'}, 'ablation': {'name':'none'}, 'mode':'retrieval-only','timestamp':123,'ingestion_results': None},
        'results': { 'metadata': {'avg_recall_at_k_10': 0.9, 'avg_retrieval_latency_p50_ms': 60}, 'easy': [] }
    }
    with open(run_dir / 'benchmark_run.json', 'w', encoding='utf-8') as f:
        json.dump(run_json, f)

    return results_dir


def test_plot_results(tmp_path):
    results_dir = create_fake_results(tmp_path)
    output_dir = tmp_path / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the plot script via subprocess (ensures environment similar to CLI)
    cmd = [
        'python', 'scripts/plot_results.py',
        '--results-dir', str(results_dir),
        '--output-dir', str(output_dir)
    ]

    subprocess.run(cmd, check=True)

    # Check some expected PNGs exist
    assert (output_dir / 'win_rate_by_domain.png').exists()
    assert (output_dir / 'recall_vs_k_summary.png').exists() or (output_dir / 'recall_vs_k.png').exists()
