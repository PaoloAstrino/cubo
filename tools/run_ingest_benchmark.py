#!/usr/bin/env python3
"""Run a reproducible ingestion benchmark and write results.

Design:
- Runs the indexing worker as a subprocess (tools.worker_index) with provided args.
- Monitors the worker process with psutil to record peak RSS (bytes) and wall time.
- Writes a JSON summary and captures stdout/stderr into a log file.

Usage (test):
  python tools/run_ingest_benchmark.py --corpus data/ingest/ingest_9_8gb_corpus_small.jsonl --index-dir results/ingest_9_8gb_test --limit 100 --out-json paper/appendix/ingest/ingest_9_8gb.json --log paper/appendix/ingest/ingest_9_8gb.log --test

Usage (full run):
  python tools/run_ingest_benchmark.py --corpus data/ingest/ingest_9_8gb_corpus.jsonl --index-dir results/ingest_9_8gb --embed-model "embedding-gemma-3-small" --out-json paper/appendix/ingest/ingest_9_8gb.json --log paper/appendix/ingest/ingest_9_8gb.log

"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = REPO_ROOT / 'paper' / 'appendix' / 'ingest' / 'ingest_9_8gb.json'
DEFAULT_LOG = REPO_ROOT / 'paper' / 'appendix' / 'ingest' / 'ingest_9_8gb.log'


def monitor_process(proc, poll_interval=0.25):
    """Monitor process and return peak RSS in bytes and elapsed time (s)."""
    p = psutil.Process(proc.pid)
    peak = 0
    start = time.time()
    try:
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss
                if rss > peak:
                    peak = rss
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            time.sleep(poll_interval)
        # Final check
        try:
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
        except Exception:
            pass
    except KeyboardInterrupt:
        proc.terminate()
        raise
    end = time.time()
    return peak, end - start


def run_index_worker(corpus, index_dir, embed_model=None, limit=None, batch_size=None, env=None, extra_args=None, log_path=None, test=False):
    # Build command: python -m tools.worker_index --corpus <path> --index-dir <dir> [--limit N]
    cmd = [sys.executable, '-m', 'tools.worker_index', '--corpus', str(corpus), '--index-dir', str(index_dir)]
    if limit:
        cmd += ['--limit', str(limit)]
    if embed_model:
        cmd += ['--embed-model', str(embed_model)]
    if batch_size:
        cmd += ['--batch-size', str(batch_size)]

    if extra_args:
        cmd += extra_args

    cmd_str = ' '.join(shlex.quote(x) for x in cmd)
    print(f"Running: {cmd_str}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'wb') as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            peak, elapsed = monitor_process(proc)
            # Stream output to log file while process finished
            for line in proc.stdout:
                log_f.write(line)
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
    return rc, peak, elapsed, cmd_str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--index-dir', type=Path, required=True)
    p.add_argument('--embed-model', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='Limit number of docs to index (test mode)')
    p.add_argument('--out-json', type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument('--log', type=Path, default=DEFAULT_LOG)
    p.add_argument('--test', action='store_true')
    p.add_argument('--venv-python', type=Path, default=sys.executable, help='Python binary to use for subprocess')
    return p.parse_args()


def main():
    args = parse_args()
    env = os.environ.copy()
    # Ensure PYTHONPATH so worker can import evaluation.beir_adapter
    env['PYTHONPATH'] = str(REPO_ROOT)

    # Prepare index dir
    args.index_dir.mkdir(parents=True, exist_ok=True)

    # Run worker
    start_ts = datetime.utcnow().isoformat() + 'Z'

    rc, peak_rss, elapsed, cmd_str = run_index_worker(
        args.corpus, args.index_dir, embed_model=args.embed_model, limit=args.limit,
        batch_size=args.batch_size, env=env, log_path=args.log, test=args.test
    )

    result = {
        'timestamp': start_ts,
        'corpus': str(args.corpus),
        'index_dir': str(args.index_dir),
        'cmd': cmd_str,
        'exit_code': rc,
        'time_s': elapsed,
        'peak_rss_bytes': peak_rss,
        'peak_rss_gb': round(peak_rss / (1024 ** 3), 3)
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as jf:
        json.dump(result, jf, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Wrote results to {args.out_json}")
    print(f"Log file: {args.log}")

    # Write a small README entry in same dir pointing to log and json
    meta_readme = args.out_json.parent / 'README.md'
    with open(meta_readme, 'w', encoding='utf-8') as rm:
        rm.write('# Ingest benchmark artifacts\n')
        rm.write('\n')
        rm.write(f'json: {args.out_json.name}\n')
        rm.write(f'log: {args.log.name}\n')
        rm.write('\n')
        rm.write('To reproduce: activate venv and run the command above (captured in `cmd` field of the JSON).')

    if rc != 0:
        print(f"Ingest worker exited with code {rc}; see log for details.")
        sys.exit(rc)

if __name__ == '__main__':
    main()
