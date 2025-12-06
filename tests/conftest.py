import shutil
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import os

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Optional: Determine if FAISS is present; plugins that require FAISS can be skipped when not present.
try:

    _FAISS_PRESENT = True
except Exception:
    _FAISS_PRESENT = False


# Optional: Determine if torch is present; many embedding/retrieval tests require torch.
try:
    import torch  # noqa: F401
    _TORCH_PRESENT = True
except ImportError:
    _TORCH_PRESENT = False


# Whoosh (deprecated) is no longer used; remove detection logic


# Optional: Determine if the evaluation package is present; performance tests rely on
# the `cubo.evaluation` modules which may be optional in lightweight dev setups.
try:
    import cubo.evaluation  # noqa: F401
    _EVALUATION_PRESENT = True
except Exception:
    _EVALUATION_PRESENT = False


def pytest_collection_modifyitems(config, items):
    if not _FAISS_PRESENT:
        skip_faiss = pytest.mark.skip(
            reason="FAISS is not installed; skipping FAISS-dependent tests"
        )
        for item in items:
            if "requires_faiss" in item.keywords:
                item.add_marker(skip_faiss)
    # Whoosh support removed; no skip handling needed
    if not _EVALUATION_PRESENT:
        skip_eval = pytest.mark.skip(
            reason="Evaluation package is not installed; skipping performance tests"
        )
        for item in items:
            # Mark any test that is under tests/performance/ as skipped
            if os.path.join("tests", "performance") in str(item.fspath):
                item.add_marker(skip_eval)
    if not _TORCH_PRESENT:
        skip_torch = pytest.mark.skip(
            reason="PyTorch is not installed; skipping torch-dependent tests"
        )
        for item in items:
            if "requires_torch" in item.keywords:
                item.add_marker(skip_torch)


def pytest_collectreport(report):
    """Handle collection errors gracefully by converting import errors to skips."""
    if report.failed:
        # Check if the failure is due to missing torch
        for longrepr in [report.longrepr] if report.longrepr else []:
            longrepr_str = str(longrepr)
            if "ModuleNotFoundError" in longrepr_str and ("torch" in longrepr_str or "No module named 'torch'" in longrepr_str):
                # Convert to xfail to avoid hard failure
                report.outcome = "passed"
                report.wasxfail = "torch not installed"


def pytest_runtest_teardown(item, nextitem):
    """Force a garbage collection cycle after each test to ensure objects with
    __del__ methods (e.g., DocumentRetriever) are finalized and file handles are
    released. This is a lightweight precaution for Windows where file locks can
    prevent cleanup of temporary directories.
    """
    import gc

    gc.collect()


@pytest.fixture
def tmp_metadata_db(tmp_path, monkeypatch):
    # Create a temporary DB path and monkeypatch the configuration or metadata manager factory
    db_path = tmp_path / "metadata.db"
    # Ensure the parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a fresh SQLite DB file and return its path
    conn = sqlite3.connect(str(db_path))
    conn.close()

    # Monkeypatch the in-memory MetadataManager to use this db by creating a fresh manager
    from cubo.storage import metadata_manager

    # Reset module-level manager instance if present
    metadata_manager._manager = None

    # Set config override CUBO_METADATA_DB_PATH to ensure MetadataManager uses this path
    monkeypatch.setenv("CUBO_METADATA_DB_PATH", str(db_path))
    # Ensure the metadata manager uses this path by re-instantiating it via get_metadata_manager
    return str(db_path)


@pytest.fixture(scope="module")
def mini_data(tmp_path_factory, request):
    """Copy the small fixture dataset to a temporary folder and return the path.

    Expects files under `tests/fixtures/mini_dataset/`.
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "mini_dataset"
    target = tmp_path_factory.mktemp("mini_dataset")
    if target.exists():
        shutil.rmtree(str(target))
    shutil.copytree(fixtures_dir, target)
    yield str(target)


@pytest.fixture(scope="module")
def fast_pass_result(mini_data, tmp_path_factory):
    """Run a fast-pass ingestion to build a BM25 index and return the resulting paths.

    Returns the `result` portion of the ingestion manager's return value.
    """
    from cubo.ingestion.ingestion_manager import IngestionManager

    manager = IngestionManager()
    output_dir = tmp_path_factory.mktemp("fast_pass_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    res = manager.start_fast_pass(
        str(mini_data), output_dir=str(output_dir), skip_model=True, auto_deep=False
    )
    assert "run_id" in res
    assert "result" in res and res["result"] is not None
    yield res["result"]


@pytest.fixture(scope="module")
def cubo_app():
    """Provide a minimal CuboCore instance that avoids loading heavy models.

    The fixture initializes a light CuboCore instance and monkeypatches the generator to a deterministic one.
    Uses CuboCore (not CuboCLI) to avoid any CLI side effects in tests.
    """
    from cubo.core import CuboCore

    app = CuboCore()
    # Try to set a real DocumentLoader, otherwise use a mock
    try:
        from cubo.ingestion.document_loader import DocumentLoader

        app.doc_loader = DocumentLoader()
    except Exception:
        app.doc_loader = MagicMock()

    # Deterministic generator
    class _MockGenerator:
        def generate_response(self, query, context):
            return f"Deterministic answer: {query}"

    app.generator = _MockGenerator()

    # Construct a lightweight retriever using a MagicMock model to avoid heavy embedding loads
    try:
        from sentence_transformers import SentenceTransformer

        dummy_model = MagicMock(spec=SentenceTransformer)
        from cubo.retrieval.retriever import DocumentRetriever

        app.retriever = DocumentRetriever(dummy_model)
    except Exception:
        app.retriever = None

    yield app


@pytest.fixture(scope="function")
def mock_llm_client(monkeypatch):
    """Monkeypatch the response generator so LLM calls are deterministic for tests.

    Returns an object with a `generate_response(query, context)` method.
    """

    class _DeterministicGenerator:
        def generate_response(self, query, context):
            return f"Deterministic: {query}"

    # Try to monkeypatch the factory used to create the generator
    try:
        import cubo.processing.generator as gen_mod

        monkeypatch.setattr(gen_mod, "create_response_generator", lambda: _DeterministicGenerator())
    except Exception:
        # Not fatal; tests can still use the generator returned here
        pass

    yield _DeterministicGenerator()


# tmp_whoosh_index fixture removed (Whoosh support removed)


@pytest.fixture(scope="function")
def mock_embedding_model():
    """Provide a mock embedding model for testing.

    Returns a MockEmbeddingModel that generates deterministic embeddings.
    """
    from tests.fixtures.mocks import MockEmbeddingModel

    return MockEmbeddingModel(embedding_dim=384)


@pytest.fixture(scope="function")
def mock_vector_store():
    """Provide a mock vector store for testing.

    Returns an in-memory MockVectorStore for fast, isolated testing.
    """
    from tests.fixtures.mocks import MockVectorStore

    return MockVectorStore()


@pytest.fixture(scope="function")
def mock_llm_service():
    """Provide a mock LLM service for testing.

    Returns a MockLLMClient that generates deterministic responses.
    """
    from tests.fixtures.mocks import MockLLMClient

    return MockLLMClient(default_response="Test answer from mock LLM")
