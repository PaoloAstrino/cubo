import pytest

from cubo.retrieval.bm25_store_factory import get_bm25_store
from cubo.retrieval.bm25_python_store import BM25PythonStore


def test_factory_returns_python_by_default(tmp_path):
    # Default backend should be Python store
    st = get_bm25_store(index_dir=str(tmp_path))
    assert isinstance(st, BM25PythonStore)


def test_factory_explicit_python():
    st = get_bm25_store("python")
    from cubo.retrieval.bm25_python_store import BM25PythonStore

    assert isinstance(st, BM25PythonStore)


def test_factory_returns_whoosh_by_default_if_available():
    st = get_bm25_store()
    assert isinstance(st, BM25PythonStore)


def test_factory_whoosh_falls_back_to_python(tmp_path):
    st = get_bm25_store("whoosh", index_dir=str(tmp_path))
    assert isinstance(st, BM25PythonStore)
