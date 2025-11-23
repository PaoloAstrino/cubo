import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import sys
import time

import pytest

from scripts import benchmark_runner as br
from scripts.benchmark_runner import BenchmarkRunner


def write_dummy_run(run_dir: Path, dataset_name: str = 'ds', config_name: str = 'config', ablation_name: str = 'none', timestamp: int = 12345):
    run_id = f"{dataset_name}__{config_name}__{ablation_name}__{int(timestamp)}"
    rdir = run_dir / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    # Write a dummy benchmark_run.json
    content = {
        'metadata': {
            'run_id': run_id,
            'dataset': {'path': 'data/smoke', 'name': dataset_name},
            'retrieval_config': {'name': config_name},
            'ablation': {'name': ablation_name},
            'mode': 'retrieval-only',
            'timestamp': timestamp
        },
        'results': {'dummy': True}
    }
    with open(rdir / 'benchmark_run.json', 'w', encoding='utf-8') as fh:
        json.dump(content, fh)
    return rdir, run_id


def test_skip_existing_run(tmp_path, monkeypatch):
    # Setup runner with a fake dataset and a single config/ablation
    datasets = [{'path': str(tmp_path), 'name': 'smoke'}]
    retrieval_configs = [{'name': 'hybrid_test', 'config_updates': {}}]
    ablations = [{'name': 'none', 'config_updates': {}}]

    outdir = tmp_path / 'results'
    outdir.mkdir(parents=True, exist_ok=True)

    # Force a deterministic timestamp for the run
    monkeypatch.setattr(br.time, 'time', lambda: 12345)

    # Create existing run dir with benchmark_run.json
    run_dir, run_id = write_dummy_run(outdir, dataset_name='smoke', config_name='hybrid_test', ablation_name='none', timestamp=12345)

    runner = BenchmarkRunner(datasets=datasets, retrieval_configs=retrieval_configs, ablations=ablations, k_values=[5], mode='retrieval-only', output_dir=str(outdir), max_retries=1, retry_backoff=0.1, skip_existing=True)

    # monkeypatch runner._run_with_retries to raise if called (should not be invoked due to skip)
    called = {'yes': False}
    def fake_run(cmd, cwd=None, max_retries=1, backoff=0.1):
        called['yes'] = True
        return True, 1, None
    monkeypatch.setattr(runner, '_run_with_retries', fake_run)

    runner.run(run_ingest_first=True)

    # Ensure that runner did not call _run_with_retries (skipped)
    assert called['yes'] is False

    # Ensure the dummy run file still exists and wasn't overwritten
    assert (run_dir / 'benchmark_run.json').exists()
