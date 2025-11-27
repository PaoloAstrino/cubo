import os
import stat
import sys

from scripts.benchmark_runner import BenchmarkRunner


def test_run_with_retries_success(tmp_path):
    # Create a small failing script that will succeed on 3rd run
    script_path = tmp_path / "failing_script.py"
    state_file = tmp_path / "state.txt"
    output_file = tmp_path / "out.json"
    threshold = 3

    script_content = f"""
import sys
from pathlib import Path
state_file = Path('{state_file.as_posix()}')
if state_file.exists():
    attempt = int(state_file.read_text().strip())
else:
    attempt = 0
attempt += 1
state_file.write_text(str(attempt))
if attempt < {threshold}:
    sys.exit(1)
else:
    import json
    with open('{output_file.as_posix()}', 'w') as f:
        json.dump({{'success': True, 'attempt': attempt}}, f)
    sys.exit(0)
"""

    script_path.write_text(script_content)
    # Make the script executable on Unix (no-op on Windows)
    try:
        st = os.stat(str(script_path))
        os.chmod(str(script_path), st.st_mode | stat.S_IEXEC)
    except Exception:
        pass

    # Create runner
    runner = BenchmarkRunner(
        datasets=[], retrieval_configs=[], ablations=[], k_values=[5], mode="retrieval-only"
    )

    cmd = [sys.executable, str(script_path)]
    succeeded, attempts, err = runner._run_with_retries(
        cmd, cwd=str(tmp_path), max_retries=5, backoff=0.1
    )
    assert succeeded is True
    assert attempts >= threshold
    assert output_file.exists()


def test_run_with_retries_fail(tmp_path):
    # Script that always exits with error
    script_path = tmp_path / "always_fail.py"
    script_path.write_text("import sys; sys.exit(2)")

    runner = BenchmarkRunner(
        datasets=[], retrieval_configs=[], ablations=[], k_values=[5], mode="retrieval-only"
    )
    cmd = [sys.executable, str(script_path)]
    succeeded, attempts, err = runner._run_with_retries(
        cmd, cwd=str(tmp_path), max_retries=2, backoff=0.1
    )
    assert succeeded is False
    assert attempts == 2
    assert err is not None
