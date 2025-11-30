import importlib
import pytest
from typing import Optional


def pytest_ignore_collect(path, config) -> Optional[bool]:
    """Ignore collecting performance tests when optional evaluation module missing.

    Using pytest_ignore_collect prevents PyTest from importing module-level code in performance
    tests, avoiding ModuleNotFoundError when optional packages are not installed.
    """
    # Only inspect files under 'tests/performance'
    if "tests{}performance".format(("/")) in str(path).replace("\\", "/"):
        try:
            spec = importlib.util.find_spec("cubo.evaluation")
        except Exception:
            spec = None
        if spec is None:
            return True
    return None
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def smoke_data_dir(tmp_path_factory):
    # Return a path to the smoke dataset (existing sample)
    repo_root = Path(__file__).resolve().parents[3]
    smoke_dir = repo_root / "data" / "smoke"
    if not smoke_dir.exists():
        pytest.skip("Smoke dataset not installed")
    return smoke_dir


@pytest.fixture
def sample_questions(smoke_data_dir):
    return smoke_data_dir / "questions.json"


@pytest.fixture
def sample_ground_truth(smoke_data_dir):
    return smoke_data_dir / "ground_truth.json"


@pytest.fixture
def sample_data_dir(smoke_data_dir):
    return smoke_data_dir


@pytest.fixture(scope="session")
def has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    # Skip integration/tests that require GPU when not available
    has_gpu_env = os.getenv("CUBO_TEST_GPU", "0") == "1"
    if not has_gpu_env:
        skip_gpu = pytest.mark.skip(reason="GPU tests skipped - set CUBO_TEST_GPU=1 to enable")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
