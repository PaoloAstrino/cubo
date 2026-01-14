#!/usr/bin/env python3
"""
Run BEIR benchmarks on multiple datasets and generate summary report.

Usage:
    python tools/run_beir_batch.py --datasets scifact fiqa arguana
    python tools/run_beir_batch.py --all-small  # Run on small datasets only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
import time
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Dataset metadata
DATASET_INFO = {
    "nfcorpus": {"size": "small", "domain": "medical", "docs": 3633, "queries": 323},
    "scifact": {"size": "small", "domain": "scientific", "docs": 5183, "queries": 300},
    "arguana": {"size": "small", "domain": "argument", "docs": 8674, "queries": 1406},
    "fiqa": {"size": "medium", "domain": "financial", "docs": 57638, "queries": 648},
    "trec-covid": {"size": "large", "domain": "biomedical", "docs": 171332, "queries": 50},
    "webis-touche2020": {"size": "large", "domain": "argument", "docs": 382545, "queries": 49},
    "scidocs": {"size": "medium", "domain": "scientific", "docs": 25657, "queries": 1000},
    "fever": {"size": "large", "domain": "fact-checking", "docs": 5416568, "queries": 6666},
    "climate-fever": {"size": "medium", "domain": "climate", "docs": 5416593, "queries": 1535},
    "dbpedia-entity": {"size": "large", "domain": "entity", "docs": 4635922, "queries": 400},
    "nq": {"size": "xlarge", "domain": "qa", "docs": 2681468, "queries": 3452},
    "hotpotqa": {"size": "xlarge", "domain": "qa", "docs": 5233329, "queries": 7405},
    "quora": {"size": "medium", "domain": "duplicate-detection", "docs": 522931, "queries": 10000},
}


def _validate_dataset(dataset_name: str) -> tuple:
    """Validate dataset exists and return file paths."""
    dataset_dir = Path(f"data/beir/{dataset_name}")
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return None, None, None, None

    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / "test.tsv"

    if not corpus_path.exists() or not queries_path.exists():
        print(f"Missing required files in {dataset_dir}")
        return None, None, None, None
    
    return corpus_path, queries_path, qrels_path, dataset_dir


def _build_benchmark_command(dataset_name: str, corpus_path: Path, queries_path: Path, 
                             index_dir: str, output_file: str, laptop_mode: bool, 
                             no_reindex: bool) -> list:
    """Build the command to run BEIR adapter."""
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).parent / "run_beir_adapter.py"),
        "--corpus",
        str(corpus_path),
        "--queries",
        str(queries_path),
        "--output",
        output_file,
        "--index-dir",
        index_dir,
        "--use-optimized",
    ]

    if not no_reindex:
        cmd.insert(-2, "--reindex")

    if laptop_mode:
        cmd.append("--laptop-mode")

    print(f"Indexing behavior: {'reindex' if not no_reindex else 'no-reindex (use existing indexes)'}")
    print(f"Command: {' '.join(cmd)}")
    
    return cmd


def _stream_process_output(process, dataset_name: str, logfile: Path):
    """Stream subprocess output with heartbeat monitoring."""
    last_output = time.time()
    heartbeat_interval = 30  # seconds
    
    while True:
        line = process.stdout.readline()
        if line:
            last_output = time.time()
            logfile.write(line)
            logfile.flush()
            print(f"[{dataset_name}] {line.rstrip()}", flush=True)
        else:
            if process.poll() is not None:
                break
            
            if time.time() - last_output > heartbeat_interval:
                print(f"[{dataset_name}] ...still running (no recent output). Check {logfile.name} for detailed logs", flush=True)
                logfile.write(f"[heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                logfile.flush()
                last_output = time.time()
            
            time.sleep(0.5)


def _run_subprocess(cmd: list, dataset_name: str, logfile: Path) -> bool:
    """Run subprocess and stream output, return success status."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        env=env
    )
    
    try:
        _stream_process_output(process, dataset_name, logfile)
        rc = process.wait()
        return rc == 0
    except Exception as e:
        process.kill()
        print(f"ERROR: Benchmark crashed: {e}. Check partial logs at {logfile}", flush=True)
        return False


def _calculate_dataset_metrics(dataset_name: str, output_file: str, qrels_path: Path) -> dict:
    """Calculate metrics for the dataset."""
    if not qrels_path.exists():
        print(f"Warning: No qrels found at {qrels_path}, skipping metrics")
        return {"dataset": dataset_name, "status": "no_qrels"}

    print("Calculating metrics...")
    try:
        import io
        from contextlib import redirect_stdout
        from tools.calculate_beir_metrics import calculate_metrics

        f = io.StringIO()
        with redirect_stdout(f):
            calculate_metrics(output_file, str(qrels_path), k=10)

        output = f.getvalue()

        metrics = {"dataset": dataset_name, "status": "success"}
        for line in output.split("\n"):
            if "Queries Evaluated:" in line:
                metrics["queries_evaluated"] = int(line.split(":")[-1].strip())
            elif "Avg Recall@10:" in line:
                metrics["recall_at_10"] = float(line.split(":")[-1].strip())
            elif "Mean Reciprocal Rank:" in line:
                metrics["mrr"] = float(line.split(":")[-1].strip())

        return metrics
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return {"dataset": dataset_name, "status": "metrics_failed", "error": str(e)}


def run_benchmark(dataset_name: str, laptop_mode: bool = True, no_reindex: bool = False) -> dict:
    """Run benchmark on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Running benchmark on: {dataset_name}")
    print(f"{'='*60}")

    # Validate dataset
    corpus_path, queries_path, qrels_path, dataset_dir = _validate_dataset(dataset_name)
    if not corpus_path:
        return None

    # Setup paths
    output_file = f"results/beir_run_{dataset_name}.json"
    index_dir = f"results/beir_index_{dataset_name}"

    # Build command
    cmd = _build_benchmark_command(
        dataset_name, corpus_path, queries_path, index_dir, 
        output_file, laptop_mode, no_reindex
    )

    # Setup logging
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = logs_dir / f"beir_{dataset_name}.log"
    print(f"Streaming benchmark logs to: {logfile_path}", flush=True)

    # Run subprocess
    start_time = time.time()
    with open(logfile_path, "w", encoding="utf-8") as logfile:
        success = _run_subprocess(cmd, dataset_name, logfile)
    
    elapsed = time.time() - start_time
    
    if not success:
        print(f"ERROR: Benchmark failed — see {logfile_path}", flush=True)
        return None
    
    print(f"OK: Benchmark completed in {elapsed:.1f}s (logs: {logfile_path})", flush=True)

    # Calculate metrics
    return _calculate_dataset_metrics(dataset_name, output_file, qrels_path)


def generate_summary_markdown(results: list, output_path: Path):
    """Generate markdown summary of all results."""
    md = ["# BEIR Multi-Dataset Benchmark Results\n"]
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"**Model**: `embeddinggemma-300m`\n")
    md.append(f"**Configuration**: Laptop Mode (optimized batch retrieval, no reranking)\n")

    md.append("\n## Results Summary\n")
    md.append("| Dataset | Domain | Size | Queries | Recall@10 | MRR | Status |")
    md.append("|---------|--------|------|---------|-----------|-----|--------|")

    for result in results:
        dataset = result.get("dataset", "unknown")
        info = DATASET_INFO.get(dataset, {})
        domain = info.get("domain", "unknown")
        size = info.get("size", "unknown")
        queries = result.get("queries_evaluated", "N/A")
        recall = f"{result.get('recall_at_10', 0):.4f}" if "recall_at_10" in result else "N/A"
        mrr = f"{result.get('mrr', 0):.4f}" if "mrr" in result else "N/A"
        status = result.get("status", "unknown")

        md.append(f"| {dataset} | {domain} | {size} | {queries} | {recall} | {mrr} | {status} |")

    md.append("\n## Analysis\n")

    # Group by domain
    by_domain = {}
    for result in results:
        if "recall_at_10" in result:
            dataset = result["dataset"]
            domain = DATASET_INFO.get(dataset, {}).get("domain", "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result)

    md.append("### Performance by Domain\n")
    for domain, domain_results in sorted(by_domain.items()):
        avg_recall = sum(r["recall_at_10"] for r in domain_results) / len(domain_results)
        avg_mrr = sum(r["mrr"] for r in domain_results) / len(domain_results)
        md.append(
            f"- **{domain.capitalize()}**: Avg Recall@10 = {avg_recall:.4f}, Avg MRR = {avg_mrr:.4f}"
        )

    md.append("\n### Observations\n")
    md.append("- Add your observations here based on the results\n")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n✓ Summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run BEIR benchmarks on multiple datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="List of dataset names to benchmark",
    )
    parser.add_argument(
        "--all-small",
        action="store_true",
        help="Run on all small datasets",
    )
    parser.add_argument(
        "--all-medium",
        action="store_true",
        help="Run on all small and medium datasets",
    )
    parser.add_argument(
        "--no-laptop-mode",
        action="store_true",
        help="Disable laptop mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/beir_multi_dataset_summary.md",
        help="Output markdown file",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Do not rebuild indexes; use existing indexes if present",
    )

    args = parser.parse_args()

    datasets_to_run = []

    if args.all_small:
        datasets_to_run = [name for name, info in DATASET_INFO.items() if info["size"] == "small"]
    elif args.all_medium:
        datasets_to_run = [
            name for name, info in DATASET_INFO.items() if info["size"] in ["small", "medium"]
        ]
    elif args.datasets:
        datasets_to_run = args.datasets
    else:
        parser.print_help()
        return

    print(f"Running benchmarks on {len(datasets_to_run)} datasets: {datasets_to_run}")

    results = []
    for dataset_name in datasets_to_run:
        result = run_benchmark(dataset_name, laptop_mode=not args.no_laptop_mode, no_reindex=args.no_reindex)
        if result:
            results.append(result)
            # Save intermediate results
            with open("results/beir_batch_results.json", "w") as f:
                json.dump(results, f, indent=2)

    # Generate summary
    generate_summary_markdown(results, Path(args.output))

    print(f"\n{'='*60}")
    print(f"Completed {len(results)}/{len(datasets_to_run)} benchmarks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
