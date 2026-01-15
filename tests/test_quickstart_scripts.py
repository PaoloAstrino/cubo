import stat
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_scripts_exist():
    sh = REPO_ROOT / "run_local.sh"
    ps1 = REPO_ROOT / "run_local.ps1"
    assert sh.exists(), "run_local.sh must exist"
    assert ps1.exists(), "run_local.ps1 must exist"

    # Both scripts should mention the main startup script
    content_sh = sh.read_text(encoding="utf-8")
    content_ps1 = ps1.read_text(encoding="utf-8")
    assert "start_fullstack.py" in content_sh, "run_local.sh should start start_fullstack.py"
    assert "start_fullstack.py" in content_ps1, "run_local.ps1 should start start_fullstack.py"

    # Ensure the canonical run_local.sh does an editable install and installs frontend deps
    assert "install -e ." in content_sh, "run_local.sh should install the package in editable mode (pip install -e .)"
    assert ("pnpm" in content_sh) or ("npm" in content_sh), "run_local.sh should install frontend dependencies with pnpm or npm"

    # Ensure quiet defaults exist (-q / --silent) so the script is unobtrusive by default
    assert "-q" in content_sh or "--silent" in content_sh

    # Preflight checks
    assert "Checking Python version" in content_sh
    assert "Checking available disk space" in content_sh
    assert "Checking Node" in content_sh or "Node.js" in content_sh

    content_ps1 = ps1.read_text(encoding="utf-8")
    assert "Checking Python version" in content_ps1
    assert "Checking available disk space" in content_ps1
    assert "Node.js" in content_ps1

    # PowerShell script should accept -Verbose switch or reference CUBO_VERBOSE
    assert "param(" in content_ps1
    assert "Verbose" in content_ps1 or "CUBO_VERBOSE" in content_ps1


def test_run_local_sh_executable():
    sh = REPO_ROOT / "run_local.sh"

    # On Windows, executable bits are not meaningful in the same way, so skip that check
    if sys.platform.startswith("win"):
        pytest.skip("Executable bit check skipped on Windows")

    st = sh.stat()
    is_executable = bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    assert is_executable, "run_local.sh should be executable (chmod +x)"
