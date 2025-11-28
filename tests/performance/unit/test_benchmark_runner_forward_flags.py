import json
import os
from pathlib import Path

from scripts.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_forwards_auto_populate_flag(tmp_path):
    base_dir = str(tmp_path)
    data_folder = os.path.join(base_dir, "data")
    os.makedirs(data_folder, exist_ok=True)
    questions_path = os.path.join(base_dir, "questions.json")
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump({"questions": {}}, f)

    runner = BenchmarkRunner(
        datasets=[{"path": data_folder, "name": "sample", "questions": questions_path}],
        retrieval_configs=[{"name": "rc", "config_updates": {}}],
        ablations=[{"name": "none", "config_updates": {}}],
        k_values=[5],
        mode="retrieval-only",
        skip_index=True,
        auto_populate_db=True,
        output_dir=os.path.join(base_dir, "results"),
    )

    captured = {"cmds": []}

    def fake_run(cmd, cwd=None, max_retries=3, backoff=2.0, stdout_path=None, stderr_path=None):
        captured["cmds"].append(cmd)
        return True, 1, None

    runner._run_with_retries = fake_run
    runner.run(run_ingest_first=False)

    # Ensure we captured at least one subprocess command and that run_rag_tests was invoked with --auto-populate-db
    assert len(captured["cmds"]) > 0
    found = False
    for cmd in captured["cmds"]:
        if isinstance(cmd, list) and "scripts/run_rag_tests.py" in cmd:
            assert "--skip-index" in cmd
            assert "--auto-populate-db" in cmd
            found = True
            break
    assert found


def test_benchmark_runner_does_not_forward_auto_populate_when_false(tmp_path):
    base_dir = str(tmp_path)
    data_folder = os.path.join(base_dir, "data")
    os.makedirs(data_folder, exist_ok=True)
    questions_path = os.path.join(base_dir, "questions.json")
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump({"questions": {}}, f)

    runner = BenchmarkRunner(
        datasets=[{"path": data_folder, "name": "sample", "questions": questions_path}],
        retrieval_configs=[{"name": "rc", "config_updates": {}}],
        ablations=[{"name": "none", "config_updates": {}}],
        k_values=[5],
        mode="retrieval-only",
        skip_index=True,
        auto_populate_db=False,
        output_dir=os.path.join(base_dir, "results"),
    )

    captured = {"cmds": []}

    def fake_run(cmd, cwd=None, max_retries=3, backoff=2.0, stdout_path=None, stderr_path=None):
        captured["cmds"].append(cmd)
        return True, 1, None

    runner._run_with_retries = fake_run
    runner.run(run_ingest_first=False)

    found_forward_flag = False
    for cmd in captured["cmds"]:
        if isinstance(cmd, list) and "scripts/run_rag_tests.py" in cmd:
            if "--auto-populate-db" in cmd:
                found_forward_flag = True
                break
    assert not found_forward_flag
