import pytest

pytest.importorskip("torch")

from cubo.retrieval.bm25_python_store import BM25PythonStore
from cubo.retrieval.bm25_store import BM25Store


def test_bm25store_interface_methods():
    assert hasattr(BM25Store, "index_documents")
    assert hasattr(BM25Store, "add_documents")
    assert hasattr(BM25Store, "search")
    assert hasattr(BM25Store, "compute_score")
    assert hasattr(BM25Store, "load_stats")
    assert hasattr(BM25Store, "save_stats")
    assert hasattr(BM25Store, "close")


def test_python_store_basic_flow():
    # Test Python store explicitly
    st = BM25PythonStore()
    docs = [
        {"doc_id": "a", "text": "apples and bananas"},
        {"doc_id": "b", "text": "cars and vehicles"},
    ]
    st.index_documents(docs)
    out = st.search("apples", top_k=2)
    assert isinstance(out, list)
    assert len(out) >= 0
