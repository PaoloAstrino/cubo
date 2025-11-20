import sys
from pathlib import Path
import pytest

# Ensure repo root (project root) is in sys.path so tests can import src.* modules
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Optional: Determine if FAISS is present; plugins that require FAISS can be skipped when not present.
try:
    import faiss  # type: ignore
    _FAISS_PRESENT = True
except Exception:
    _FAISS_PRESENT = False


def pytest_collection_modifyitems(config, items):
    if not _FAISS_PRESENT:
        skip_faiss = pytest.mark.skip(reason="FAISS is not installed; skipping FAISS-dependent tests")
        for item in items:
            # convention: tests that require FAISS use 'requires_faiss' keyword marker
            if "requires_faiss" in item.keywords:
                item.add_marker(skip_faiss)
