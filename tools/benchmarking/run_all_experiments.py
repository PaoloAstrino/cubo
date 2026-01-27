"""Sequential experiment runner (Option B) — Python implementation.

- Runs M1-M4 sequentially (safe for 16GB machines)
- Writes per-step logs to results/logs/, JSON outputs to results/
- Writes a run manifest to results/manifests/

Run:
    python tools/run_all_experiments.py &

The script is resilient: it records exit codes and continues to the next step.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
MANIFESTS = RESULTS / "manifests"

RESULTS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
MANIFESTS.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
try:
    GIT_SHA = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    )
except Exception:
    GIT_SHA = "unknown"

MANIFEST_PATH = MANIFESTS / f"{RUN_ID}-{GIT_SHA}-manifest.json"


def write_manifest(man: Dict):
    MANIFEST_PATH.write_text(json.dumps(man, indent=2, ensure_ascii=False))


def run_step(
    name: str, cmd: List[str], output_json: Path | None, timeout: int | None = None
) -> Dict:
    log_path = LOGS / f"{RUN_ID}-{name.replace(' ', '_')}.log"
    print(f"[STEP] {name} -> logging to {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("wb") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    entry = {
        "name": name,
        "cmd": cmd,
        "exit_code": proc.returncode,
        "log": str(log_path),
        "output": str(output_json) if output_json and output_json.exists() else None,
        "start_time": None,
        "end_time": None,
    }
    return entry


def seq_run():
    manifest = {
        "run_id": RUN_ID,
        "git_sha": GIT_SHA,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "host": subprocess.check_output(["hostname"]).decode().strip(),
        "python": sys.version.splitlines()[0],
        "steps": [],
    }
    write_manifest(manifest)

    # Prefer canonical BEIR SciFact test index when available
    index_dir = (
        "data/beir_index_scifact"
        if (ROOT / "data" / "beir_index_scifact").exists()
        else "data/faiss_test"
    )
    baseline_dir = RESULTS / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "M1_concurrency",
            [
                sys.executable,
                "tools/benchmark_concurrency_real.py",
                "--index-dir",
                index_dir,
                "--num-workers",
                "4",
                "--queries-per-worker",
                "100",
                "--concurrent-ingestion",
                "--ingestion-docs",
                "data/smoke",
                "--output",
                str(RESULTS / "concurrency.json"),
            ],
            RESULTS / "concurrency.json",
            None,
        ),
        (
            "M2_retrieval_breakdown",
            [
                sys.executable,
                "tools/profile_retrieval_breakdown_real.py",
                "--index-dir",
                index_dir,
                "--queries",
                "data/beir/scifact/queries.jsonl",
                "--num-samples",
                "200",
                "--output",
                str(RESULTS / "breakdown.json"),
            ],
            RESULTS / "breakdown.json",
            None,
        ),
        (
            "M2_sensitivity",
            [
                sys.executable,
                "tools/sensitivity_analysis_real.py",
                "--index-dir",
                index_dir,
                "--queries",
                "data/beir/scifact/queries.jsonl",
                "--nprobe-values",
                "1,5,10,20,50",
                "--num-samples",
                "50",
                "--output",
                str(RESULTS / "sensitivity.json"),
            ],
            RESULTS / "sensitivity.json",
            None,
        ),
        (
            "M3_baseline_bm25",
            [
                sys.executable,
                "tools/run_pyserini_baseline.py",
                "--dataset",
                "scifact",
                "--data-dir",
                "data/beir",
                "--memory-limit",
                "16GB",
                "--output",
                str(baseline_dir / "scifact.bm25.json"),
            ],
            baseline_dir / "scifact.bm25.json",
            None,
        ),
        (
            "M3_baseline_splade",
            [
                sys.executable,
                "tools/run_splade_baseline.py",
                "--dataset",
                "scifact",
                "--data-dir",
                "data/beir",
                "--cpu-only",
                "--max-memory",
                "15GB",
                "--output",
                str(baseline_dir / "scifact.splade.json"),
            ],
            baseline_dir / "scifact.splade.json",
            None,
        ),
        (
            "M3_baseline_e5",
            [
                sys.executable,
                "tools/run_e5_ivfpq_baseline.py",
                "--dataset",
                "scifact",
                "--data-dir",
                "data/beir",
                "--memory-limit",
                "16GB",
                "--output",
                str(baseline_dir / "scifact.e5.json"),
            ],
            baseline_dir / "scifact.e5.json",
            None,
        ),
        (
            "M4_miracl_de_with",
            [
                sys.executable,
                "tools/run_multilingual_eval.py",
                "--dataset",
                "miracl-de",
                "--queries",
                "data/multilingual/miracl-de/queries.jsonl",
                "--use-compound-splitter",
                "--index-dir",
                "data/legal_de",
                "--output",
                str(RESULTS / "multilingual_with.json"),
            ],
            RESULTS / "multilingual_with.json",
            None,
        ),
        (
            "M4_miracl_de_without",
            [
                sys.executable,
                "tools/run_multilingual_eval.py",
                "--dataset",
                "miracl-de",
                "--queries",
                "data/multilingual/miracl-de/queries.jsonl",
                "--index-dir",
                "data/legal_de",
                "--output",
                str(RESULTS / "multilingual_without.json"),
            ],
            RESULTS / "multilingual_without.json",
            None,
        ),
    ]

    for name, cmd, out_json, timeout in steps:
        start = datetime.utcnow()
        entry = run_step(name, cmd, out_json, timeout)
        entry["start_time"] = start.isoformat() + "Z"
        entry["end_time"] = datetime.utcnow().isoformat() + "Z"
        manifest["steps"].append(entry)
        write_manifest(manifest)

    manifest["end_time"] = datetime.utcnow().isoformat() + "Z"
    write_manifest(manifest)
    print(f"[DONE] Sequential run finished — manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    seq_run()
