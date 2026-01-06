import os
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


def test_run_local_sh_executable():
    sh = REPO_ROOT / "run_local.sh"

    # On Windows, executable bits are not meaningful in the same way, so skip that check
    if sys.platform.startswith("win"):
        pytest.skip("Executable bit check skipped on Windows")

    st = sh.stat()
    is_executable = bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    assert is_executable, "run_local.sh should be executable (chmod +x)"