import pytest
from src.cubo.retrieval.bm25_store_factory import get_bm25_store

try:
    from whoosh import index as whoosh_index
    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False


def test_factory_returns_python_by_default():
    st = get_bm25_store()
    from src.cubo.retrieval.bm25_python_store import BM25PythonStore
    assert isinstance(st, BM25PythonStore)


def test_factory_explicit_python():
    st = get_bm25_store('python')
    from src.cubo.retrieval.bm25_python_store import BM25PythonStore
    assert isinstance(st, BM25PythonStore)


@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason='Whoosh not installed')
def test_factory_whoosh(tmp_path):
    st = get_bm25_store('whoosh', index_dir=str(tmp_path))
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
    assert isinstance(st, BM25WhooshStore)
