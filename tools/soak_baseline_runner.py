"""Run repeated baseline runs for long soak testing with memory watchdog.

Example:
    python tools/soak_baseline_runner.py \
        --cmd "python tools/run_splade_baseline.py --dataset scifact --data-dir data/beir --cpu-only --max-memory 15GB --output results/baselines/scifact.splade.iter.json" \
        --iterations 200 \
        --memory-limit-gb 15 \
        --out results/baselines/scifact.splade.long.json \
        --log results/logs/splade_soak.log

The script:
- Runs the provided command repeatedly (iterations or duration)
- Monitors process and system RSS; kills run if memory_limit exceeded
- Appends per-run metadata and exit codes into an aggregated JSON
- Writes incremental state to the output file so partial results are preserved
"""
from __future__ import annotations
import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import psutil

ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def run_single(cmd: List[str], timeout: int | None, memory_limit_gb: float, log_path: Path) -> Dict[str, Any]:
    start = datetime.utcnow()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    entry = {
        "cmd": cmd,
        "start_time": start.isoformat() + "Z",
        "exit_code": None,
        "end_time": None,
        "duration_s": None,
        "killed_for_memory": False,
        "log": str(log_path),
    }

    with log_path.open("ab") as f:
        try:
            while True:
                if proc.poll() is not None:
                    break
                # stream stdout
                if proc.stdout is not None:
                    chunk = proc.stdout.read1(65536)
                    if chunk:
                        f.write(chunk)
                        f.flush()
                # memory watchdog
                try:
                    p = psutil.Process(proc.pid)
                    rss = p.memory_info().rss / (1024 ** 3)
                    if rss > memory_limit_gb * 0.98:
                        p.kill()
                        entry["killed_for_memory"] = True
                        break
                except psutil.NoSuchProcess:
                    break
                # timeout
                if timeout and (datetime.utcnow() - start).total_seconds() > timeout:
                    proc.kill()
                    break
                time.sleep(0.25)
        except Exception as e:
            proc.kill()
            f.write(f"\n[soak-runner] exception: {e}\n".encode())

    end = datetime.utcnow()
    entry["end_time"] = end.isoformat() + "Z"
    entry["duration_s"] = (end - start).total_seconds()
    entry["exit_code"] = proc.returncode
    return entry


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", required=True, help="Command to run (shell-quoted) or JSON array")
    p.add_argument("--iterations", type=int, default=100, help="Number of iterations to run")
    p.add_argument("--duration-hours", type=float, default=0.0, help="Alternate: run for this many hours")
    p.add_argument("--memory-limit-gb", type=float, default=15.0)
    p.add_argument("--timeout", type=int, default=0, help="Per-run timeout in seconds (0 = none)")
    p.add_argument("--out", type=Path, required=True, help="Aggregated output JSON")
    p.add_argument("--log", type=Path, required=True, help="Log file (appended) for all runs")
    args = p.parse_args(argv)

    # Parse command
    try:
        if args.cmd.strip().startswith("["):
            cmd = json.loads(args.cmd)
        else:
            cmd = shlex.split(args.cmd)
    except Exception:
        cmd = shlex.split(args.cmd)

    results: List[Dict[str, Any]] = []
    out_path = args.out
    log_path = args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = None
    if args.duration_hours > 0:
        deadline = datetime.utcnow() + timedelta(hours=args.duration_hours)

    iteration = 0
    try:
        while True:
            iteration += 1
            tag = f"iter_{iteration:04d}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            run_log = log_path.with_name(log_path.stem + f".{tag}.log")
            meta = run_single(cmd, timeout=(args.timeout or None), memory_limit_gb=args.memory_limit_gb, log_path=run_log)
            meta["iteration"] = iteration
            meta["tag"] = tag
            results.append(meta)

            # write incremental output
            out_path.write_text(json.dumps({"run_id": now_iso(), "cmd": cmd, "results": results}, indent=2))

            # rotate main log (append run_log)
            with run_log.open('rb') as rf, log_path.open('ab') as lf:
                lf.write(b"\n===== RUN: " + tag.encode() + b" =====\n")
                lf.write(rf.read())

            # Stopping conditions
            if deadline and datetime.utcnow() >= deadline:
                break
            if args.iterations and iteration >= args.iterations:
                break
            # short sleep between runs
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("Interrupted by user")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
