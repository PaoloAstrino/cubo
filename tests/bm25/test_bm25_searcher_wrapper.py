from src.cubo.retrieval.bm25_python_store import BM25PythonStore
from src.cubo.retrieval.bm25_searcher import BM25Searcher


def test_searcher_explicit_python_backend():
    # Explicitly use Python backend for testing
    bs = BM25Searcher(backend="python")
    assert isinstance(bs._store, BM25PythonStore)
    docs = [{"doc_id": "a", "text": "apples and fruit"}, {"doc_id": "b", "text": "cars and trucks"}]
    bs.index_documents(docs)
    res = bs.search("apples", top_k=2)
    assert len(res) == 1 and res[0]["doc_id"] == "a"


def test_searcher_uses_backend_arg():
    # If explicit backend argument is given, ensure it selects it
    bs = BM25Searcher(backend="python")
    assert hasattr(bs, "_store")
    docs = [{"doc_id": "a", "text": "apples and fruit"}, {"doc_id": "b", "text": "cars and trucks"}]
    bs.index_documents(docs)
    res = bs.search("cars", top_k=2)
    assert len(res) == 1 and res[0]["doc_id"] == "b"


def test_searcher_delegates_to_default_store():
    # Default backend should be Whoosh when available, otherwise Python
    bs = BM25Searcher()
    try:
        from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

        assert isinstance(bs._store, BM25WhooshStore)
    except Exception:
        assert isinstance(bs._store, BM25PythonStore)
