import sys
from pathlib import Path
import sqlite3
import pytest
import shutil
import os
from unittest.mock import MagicMock

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


@pytest.fixture(scope="module")
def mini_data(tmp_path_factory, request):
    """Copy the small fixture dataset to a temporary folder and return the path.

    Expects files under `tests/fixtures/mini_dataset/`.
    """
    fixtures_dir = Path(__file__).parent / 'fixtures' / 'mini_dataset'
    target = tmp_path_factory.mktemp('mini_dataset')
    if target.exists():
        shutil.rmtree(str(target))
    shutil.copytree(fixtures_dir, target)
    yield str(target)


@pytest.fixture(scope="module")
def fast_pass_result(mini_data, tmp_path_factory):
    """Run a fast-pass ingestion to build a BM25 index and return the resulting paths.

    Returns the `result` portion of the ingestion manager's return value.
    """
    from src.cubo.ingestion.ingestion_manager import IngestionManager
    manager = IngestionManager()
    output_dir = tmp_path_factory.mktemp('fast_pass_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    res = manager.start_fast_pass(str(mini_data), output_dir=str(output_dir), skip_model=True, auto_deep=False)
    assert 'run_id' in res
    assert 'result' in res and res['result'] is not None
    yield res['result']


@pytest.fixture(scope='module')
def cubo_app():
    """Provide a minimal CUBOApp instance that avoids loading heavy models.

    The fixture initializes a light CUBOApp instance and monkeypatches the generator to a deterministic one.
    """
    from src.cubo.main import CUBOApp
    from unittest.mock import MagicMock
    app = CUBOApp()
    # Try to set a real DocumentLoader, otherwise use a mock
    try:
        from src.cubo.ingestion.document_loader import DocumentLoader
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
        from src.cubo.retrieval.retriever import DocumentRetriever
        app.retriever = DocumentRetriever(dummy_model)
    except Exception:
        app.retriever = None

    yield app


@pytest.fixture(scope='function')
def mock_llm_client(monkeypatch):
    """Monkeypatch the response generator so LLM calls are deterministic for tests.

    Returns an object with a `generate_response(query, context)` method.
    """
    class _DeterministicGenerator:
        def generate_response(self, query, context):
            return f"Deterministic: {query}"

    # Try to monkeypatch the factory used to create the generator
    try:
        import src.cubo.processing.generator as gen_mod
        monkeypatch.setattr(gen_mod, 'create_response_generator', lambda: _DeterministicGenerator())
    except Exception:
        # Not fatal; tests can still use the generator returned here
        pass

    yield _DeterministicGenerator()


@pytest.fixture(scope='function')
def tmp_whoosh_index(tmp_path):
    """Create a temporary Whoosh index for testing.
    
    Returns the path to a temporary Whoosh index directory.
    Tests can use this to avoid relying on the global whoosh_index/ directory.
    """
    index_dir = tmp_path / "whoosh_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    yield str(index_dir)
    # Cleanup happens automatically via tmp_path


@pytest.fixture(scope='function')
def mock_embedding_model():
    """Provide a mock embedding model for testing.
    
    Returns a MockEmbeddingModel that generates deterministic embeddings.
    """
    from tests.fixtures.mocks import MockEmbeddingModel
    return MockEmbeddingModel(embedding_dim=384)


@pytest.fixture(scope='function')
def mock_vector_store():
    """Provide a mock vector store for testing.
    
    Returns an in-memory MockVectorStore for fast, isolated testing.
    """
    from tests.fixtures.mocks import MockVectorStore
    return MockVectorStore()


@pytest.fixture(scope='function')
def mock_llm_service():
    """Provide a mock LLM service for testing.
    
    Returns a MockLLMClient that generates deterministic responses.
    """
    from tests.fixtures.mocks import MockLLMClient
    return MockLLMClient(default_response="Test answer from mock LLM")
