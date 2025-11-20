import sys
from pathlib import Path
import sqlite3
import pytest

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Optional: Determine if FAISS is present; plugins that require FAISS can be skipped when not present.
try:
    import faiss  # type: ignore
    _FAISS_PRESENT = True
except Exception:
    _FAISS_PRESENT = False


# Optional: Determine if Whoosh is present; plugin tests that require Whoosh can be skipped when not present.
try:
    import whoosh  # type: ignore
    _WHOOSH_PRESENT = True
except Exception:
    _WHOOSH_PRESENT = False


def pytest_collection_modifyitems(config, items):
    if not _FAISS_PRESENT:
        skip_faiss = pytest.mark.skip(reason="FAISS is not installed; skipping FAISS-dependent tests")
        for item in items:
            if "requires_faiss" in item.keywords:
                item.add_marker(skip_faiss)
    if not _WHOOSH_PRESENT:
        skip_whoosh = pytest.mark.skip(reason="Whoosh is not installed; skipping Whoosh-dependent tests")
        for item in items:
            if "requires_whoosh" in item.keywords:
                item.add_marker(skip_whoosh)


@pytest.fixture
def tmp_metadata_db(tmp_path, monkeypatch):
    # Create a temporary DB path and monkeypatch the configuration or metadata manager factory
    db_path = tmp_path / 'metadata.db'
    # Ensure the parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a fresh SQLite DB file and return its path
    conn = sqlite3.connect(str(db_path))
    conn.close()

    # Monkeypatch the in-memory MetadataManager to use this db by creating a fresh manager
    from src.cubo.storage import metadata_manager
    # Reset module-level manager instance if present
    metadata_manager._manager = None

    # Set config override CUBO_METADATA_DB_PATH to ensure MetadataManager uses this path
    monkeypatch.setenv('CUBO_METADATA_DB_PATH', str(db_path))
    # Ensure the metadata manager uses this path by re-instantiating it via get_metadata_manager
    return str(db_path)
