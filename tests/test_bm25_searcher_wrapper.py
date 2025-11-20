from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.retrieval.bm25_python_store import BM25PythonStore


def test_searcher_delegates_to_python_store():
    # instantiate as wrapper but default backend is python
    bs = BM25Searcher()
    assert isinstance(bs._store, BM25PythonStore)
    docs = [{'doc_id': 'a', 'text': 'apples and fruit'}, {'doc_id': 'b', 'text': 'cars and trucks'}]
    bs.index_documents(docs)
    res = bs.search('apples', top_k=2)
    assert len(res) == 1 and res[0]['doc_id'] == 'a'


def test_searcher_uses_backend_arg():
    # If explicit backend argument is given, ensure it selects it
    bs = BM25Searcher(backend='python')
    assert hasattr(bs, '_store')
    docs = [{'doc_id': 'a', 'text': 'apples and fruit'}, {'doc_id': 'b', 'text': 'cars and trucks'}]
    bs.index_documents(docs)
    res = bs.search('cars', top_k=2)
    assert len(res) == 1 and res[0]['doc_id'] == 'b'
