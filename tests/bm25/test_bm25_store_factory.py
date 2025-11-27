import pytest

from src.cubo.retrieval.bm25_store_factory import get_bm25_store

try:

    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False


@pytest.mark.requires_whoosh
@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason="Whoosh not installed")
def test_factory_returns_whoosh_by_default(tmp_path):
    # Default backend is now Whoosh
    st = get_bm25_store(index_dir=str(tmp_path))
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

    assert isinstance(st, BM25WhooshStore)


def test_factory_explicit_python():
    st = get_bm25_store("python")
    from src.cubo.retrieval.bm25_python_store import BM25PythonStore

    assert isinstance(st, BM25PythonStore)


def test_factory_returns_whoosh_by_default_if_available():
    st = get_bm25_store()
    try:

        # If Whoosh is installed and usable, the default should be Whoosh
        from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

        assert isinstance(st, BM25WhooshStore)
    except Exception:
        # Whoosh not available; fallback to Python
        from src.cubo.retrieval.bm25_python_store import BM25PythonStore

        assert isinstance(st, BM25PythonStore)


@pytest.mark.requires_whoosh
@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason="Whoosh not installed")
def test_factory_whoosh(tmp_path):
    st = get_bm25_store("whoosh", index_dir=str(tmp_path))
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

    assert isinstance(st, BM25WhooshStore)
