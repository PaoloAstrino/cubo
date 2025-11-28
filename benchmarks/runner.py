#!/usr/bin/env python3
"""
Benchmark Runner for CUBO (Phase II)

Orchestrates dataset/config sweeps and runs retrieval/ingestion tests.
Saves per-run JSON results and produces a summary CSV for plotting.

Example usage:
python scripts/benchmark_runner.py --datasets data/ultradomain_small:ultradomain --configs configs/benchmark_config.json --mode retrieval-only --k-values 5,10,20 --output-dir results/benchmark_runs

"""

import argparse
import csv
import json
import logging
import os

# Ensure project root on path for imports
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

from src.cubo.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runner for parameter sweep and dataset benchmarking."""

    def __init__(
        self,
        datasets: List[Dict[str, str]],
        retrieval_configs: List[Dict[str, Any]],
        ablations: List[Dict[str, Any]],
        k_values: List[int],
        mode: str,
        output_dir: str = "results/benchmark_runs",
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        skip_existing: bool = False,
        force: bool = False,
        skip_index: bool = False,
        auto_populate_db: bool = False,
    ):
        self.datasets = datasets
        self.retrieval_configs = retrieval_configs
        self.ablations = ablations
        self.k_values = k_values
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.csv"
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.skip_existing = bool(skip_existing)
        self.force = bool(force)
        self.skip_index = bool(skip_index)
        self.auto_populate_db = bool(auto_populate_db)

    def _save_json_results(
        self, run_dir: Path, results: Dict[str, Any], filename: str = "run_results.json"
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {out_path}")

    def _append_summary(self, row: Dict[str, Any]):
        first_write = not self.summary_path.exists()
        with open(self.summary_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
            if first_write:
                writer.writeheader()
            writer.writerow(row)

    def _apply_config(self, config_updates: Dict[str, Any]):
        # Update global config (persist in-memory for the run)
        logger.info(f"Applying config updates: {config_updates}")
        config.update(config_updates)  # Config.update merges nested dicts

    def _run_with_retries(
        self,
        cmd: List[str],
        cwd: str = None,
        max_retries: int = 3,
        backoff: float = 2.0,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ):
        """Run a subprocess command with retries and linear backoff.

        Returns:
            (succeeded: bool, attempts: int, last_error: str)
        """
        attempts = 0
        last_error = None
        for attempt in range(1, int(max_retries) + 1):
            attempts = attempt
            try:
                logger.info(
                    f"Running command (attempt {attempt}/{max_retries}): {' '.join(cmd)}"
                )
                if stdout_path or stderr_path:
                    # Ensure dir exists
                    pstdout = open(stdout_path, 'a', encoding='utf-8') if stdout_path else subprocess.DEVNULL
                    pstderr = open(stderr_path, 'a', encoding='utf-8') if stderr_path else subprocess.DEVNULL
                    subprocess.run(cmd, cwd=cwd, check=True, stdout=pstdout, stderr=pstderr)
                    if stdout_path:
                        pstdout.close()
                    if stderr_path:
                        pstderr.close()
                else:
                    subprocess.run(cmd, cwd=cwd, check=True)
                return True, attempts, None
            except subprocess.CalledProcessError as e:
                last_error = str(e)
                logger.warning(f"Command failed (attempt {attempt}/{max_retries}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error running command: {e}")
                break

            # Linear backoff between retries
            if attempt < max_retries:
                wait = backoff * attempt
                logger.info(f"Retrying after {wait} seconds...")
                time.sleep(wait)

        return False, attempts, last_error

    def run(self, run_ingest_first: bool = True, timeout: int = None):
        """Run the sweep across datasets, retrieval configs, and ablations.

        Args:
            run_ingest_first: If True, runs ingestion test prior to retrieval tests (per-dataset)
            timeout: Optional max seconds per run (not enforced yet)
        """
        combos = list(product(self.datasets, self.retrieval_configs, self.ablations))
        logger.info(f"Total runs to execute: {len(combos)}")

        for ds, rc, ab in combos:
            ds_name = ds.get("name", Path(ds.get("path", "")).stem)
            rc_name = rc.get("name", "config")
            ab_name = ab.get("name", "none")

            run_id = f"{ds_name}__{rc_name}__{ab_name}__{int(time.time())}"
            run_dir = self.output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # If skip_existing and the run exists (fully completed), skip it
            existing_file = run_dir / "benchmark_run.json"
            if self.skip_existing and existing_file.exists():
                logger.info(
                    f"Skipping run {run_id} because results already exist and skip_existing=True"
                )
                continue

            # If force, clear existing run directory
            if self.force and run_dir.exists():
                import shutil

                try:
                    shutil.rmtree(run_dir)
                except Exception as e:
                    logger.warning(f"Could not remove existing run dir {run_dir}: {e}")
                run_dir.mkdir(parents=True, exist_ok=True)

            # Apply configuration changes to CUBO config
            run_config_updates = rc.get("config_updates", {})
            ablation_updates = ab.get("config_updates", {})
            combined_updates = {**run_config_updates, **ablation_updates}
            self._apply_config(combined_updates)

            # Optionally run ingestion throughput as a subprocess
            ingestion_results = None
            if run_ingest_first:
                logger.info(
                    f"Running ingestion for dataset {ds_name} (path={ds.get('path')}) as subprocess"
                )
                ingest_cmd = [
                    sys.executable,
                    "benchmarks/ingestion/throughput.py",
                    "--data-folder",
                    ds.get("path"),
                    "--output",
                    str(run_dir / "ingestion_result.json"),
                    "--fast-pass",
                ]
                succeeded, attempts, err = self._run_with_retries(
                    ingest_cmd,
                    cwd=None,
                    max_retries=self.max_retries,
                    backoff=self.retry_backoff,
                    stdout_path=str(run_dir / 'ingest_stdout.log'),
                    stderr_path=str(run_dir / 'ingest_stderr.log'),
                )
                if succeeded:
                    try:
                        with open(run_dir / "ingestion_result.json", encoding="utf-8") as f:
                            ingestion_results = json.load(f)
                    except Exception as ex:
                        logger.error(f"Failed to load ingestion_result.json: {ex}")
                        ingestion_results = {
                            "success": False,
                            "error": "result_file_missing",
                            "attempts": attempts,
                        }
                else:
                    logger.error(f"Ingestion test failed after {attempts} attempts: {err}")
                    ingestion_results = {"success": False, "error": str(err), "attempts": attempts}

            # Run retrieval or full RAG tests
            # Run retrieval/full tests as subprocess to avoid import path issues
            run_results = None
            logger.info(
                f"Running {self.mode} tests for dataset {ds_name} with config {rc_name} and ablation {ab_name} (subprocess)"
            )
            questions_path = ds.get("questions", "test_questions.json")
            ground_truth = ds.get("ground_truth", None)
            test_cmd = [
                sys.executable,
                "benchmarks/retrieval/rag_benchmark.py",
                "--questions",
                questions_path,
                "--data-folder",
                ds.get("path"),
                "--mode",
                self.mode,
                "--k-values",
                ",".join(str(k) for k in self.k_values),
                "--output",
                str(run_dir / "test_results.json"),
            ]
            if self.skip_index:
                test_cmd.append("--skip-index")
                if getattr(self, "auto_populate_db", False):
                    test_cmd.append("--auto-populate-db")
            if ground_truth:
                test_cmd += ["--ground-truth", ground_truth]
            if ds.get("easy_limit"):
                test_cmd += ["--easy-limit", str(ds.get("easy_limit"))]
            if ds.get("medium_limit"):
                test_cmd += ["--medium-limit", str(ds.get("medium_limit"))]
            if ds.get("hard_limit"):
                test_cmd += ["--hard-limit", str(ds.get("hard_limit"))]

            succeeded, attempts, err = self._run_with_retries(
                test_cmd,
                cwd=None,
                max_retries=self.max_retries,
                backoff=self.retry_backoff,
                stdout_path=str(run_dir / 'test_stdout.log'),
                stderr_path=str(run_dir / 'test_stderr.log'),
            )
            if succeeded:
                try:
                    with open(run_dir / "test_results.json", encoding="utf-8") as f:
                        run_results = json.load(f)
                except Exception as ex:
                    logger.error(f"Failed to load test_results.json: {ex}")
                    run_results = {
                        "success": False,
                        "error": "result_file_missing",
                        "attempts": attempts,
                    }
            else:
                logger.error(f"Test run failed after {attempts} attempts: {err}")
                run_results = {"success": False, "error": str(err), "attempts": attempts}

            # Ensure run_results is a dict even if the run failed early
            if run_results is None:
                run_results = {}

            # Persist metadata and outputs
            combined_results = {
                "metadata": {
                    "run_id": run_id,
                    "dataset": ds,
                    "retrieval_config": rc,
                    "ablation": ab,
                    "mode": self.mode,
                    "timestamp": time.time(),
                    "ingestion_results": ingestion_results,
                    "attempts": {
                        "ingestion_attempts": (
                            ingestion_results.get("attempts")
                            if isinstance(ingestion_results, dict)
                            else None
                        ),
                        "test_attempts": (
                            run_results.get("attempts") if isinstance(run_results, dict) else None
                        ),
                    },
                },
                "results": run_results,
            }

            self._save_json_results(run_dir, combined_results, filename="benchmark_run.json")

            # Append row to summary CSV for plotting
            ingestion_minutes_per_gb = None
            try:
                ingestion_minutes_per_gb = (
                    combined_results["metadata"]
                    .get("ingestion_results", {})
                    .get("ingestion", {})
                    .get("minutes_per_gb")
                )
            except Exception:
                ingestion_minutes_per_gb = None

            row = {
                "run_id": run_id,
                "dataset": ds_name,
                "retrieval_config": rc_name,
                "ablation": ab_name,
                "mode": self.mode,
                "timestamp": combined_results["metadata"]["timestamp"],
                "success": combined_results["results"].get("metadata", {}).get("success_rate", 0),
                "avg_recall_at_k_10": combined_results["results"]
                .get("metadata", {})
                .get("avg_recall_at_k_10", 0),
                "avg_ndcg_at_k_10": combined_results["results"]
                .get("metadata", {})
                .get("avg_ndcg_at_k_10", 0),
                "avg_retrieval_latency_p50_ms": combined_results["results"]
                .get("metadata", {})
                .get("avg_retrieval_latency_p50_ms", 0),
                "avg_answer_relevance": combined_results["results"]
                .get("metadata", {})
                .get("avg_answer_relevance", 0),
                "ingestion_minutes_per_gb": ingestion_minutes_per_gb,
            }
            # Append attempts and error message if present
            row["ingestion_attempts"] = (
                combined_results["metadata"].get("attempts", {}).get("ingestion_attempts")
            )
            row["test_attempts"] = (
                combined_results["metadata"].get("attempts", {}).get("test_attempts")
            )
            row["error_message"] = None
            if isinstance(run_results, dict) and run_results.get("error"):
                row["error_message"] = run_results.get("error")
            elif isinstance(ingestion_results, dict) and ingestion_results.get("error"):
                row["error_message"] = ingestion_results.get("error")

            self._append_summary(row)

        logger.info("All runs complete")


def load_json_configs(config_file: str) -> List[Dict[str, Any]]:
    """Load a list of configurations from a JSON file. Each config is a dict with fields 'name' and optional 'config_updates'"""
    with open(config_file, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("configs", [])


def parse_datasets(datasets_arg: List[str]) -> List[Dict[str, Any]]:
    """Parse `path:name` dataset strings into dicts"""
    parsed = []
    for ds in datasets_arg:
        if ":" in ds:
            path, name = ds.split(":", 1)
        else:
            path, name = ds, Path(ds).stem
        parsed.append({"path": path, "name": name})
    return parsed


def main():
    parser = argparse.ArgumentParser(description="CUBO Benchmark Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset path:alias pairs: data/ultradomain_small:ultradomain",
    )
    parser.add_argument(
        "--configs", required=True, help="Path to JSON config file with retrieval configs"
    )
    parser.add_argument(
        "--ablations", required=False, default=None, help="Path to JSON file with ablation configs"
    )
    parser.add_argument("--k-values", default="5,10,20", help="Comma separated K values")
    parser.add_argument(
        "--mode", default="retrieval-only", choices=["retrieval-only", "full", "ingestion-only"]
    )
    parser.add_argument("--output-dir", default="results/benchmark_runs")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed ingestion/test subprocesses",
    )
    parser.add_argument(
        "--retry-backoff", type=float, default=2.0, help="Backoff factor (seconds) between retries"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have results (benchmark_run.json)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Remove existing run directory and force re-run"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Pass --skip-index to run_rag_tests.py to avoid reindexing during test subprocesses",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Do not run ingestion subprocess before retrieval tests (assume vector store is already populated)",
    )

    parser.add_argument(
        "--auto-populate-db",
        dest="auto_populate_db",
        action="store_true",
        default=None,
        help="Automatically populate documents.db from BEIR corpus when --skip-index is used and DB is empty. Default is to auto-populate when --skip-index is used unless set with --no-auto-populate-db",
    )
    parser.add_argument(
        "--no-auto-populate-db",
        dest="auto_populate_db",
        action="store_false",
        help="Disable auto-population of documents.db when --skip-index is used",
    )

    parser.add_argument(
        "--questions",
        default=None,
        help="Optional global questions JSON path to use if datasets do not include questions",
    )
    args = parser.parse_args()
    datasets = parse_datasets(args.datasets)

    retrieval_configs = load_json_configs(args.configs)
    if args.ablations:
        ablations = load_json_configs(args.ablations)
    else:
        ablations = [{"name": "none", "config_updates": {}}]

    k_values = [int(x.strip()) for x in args.k_values.split(",")]

    runner = BenchmarkRunner(
        datasets,
        retrieval_configs,
        ablations,
        k_values,
        mode=args.mode,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        skip_existing=args.skip_existing,
        force=args.force,
        skip_index=args.skip_index,
        auto_populate_db=args.auto_populate_db,
    )
    # Default auto-populate behavior: if not explicitly set, enable when skip-index is used
    if args.auto_populate_db is None:
        runner.auto_populate_db = bool(args.skip_index)
    else:
        runner.auto_populate_db = bool(args.auto_populate_db)
    # If a global questions file provided, assign to each dataset if absent
    if args.questions:
        for ds in datasets:
            if "questions" not in ds:
                ds["questions"] = args.questions
    runner.run(run_ingest_first=not args.skip_ingest)


if __name__ == "__main__":
    main()
