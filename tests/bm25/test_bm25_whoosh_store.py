import pytest

try:
    from whoosh import index as whoosh_index
    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False

@pytest.mark.requires_whoosh
@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason='Whoosh not installed')
def test_whoosh_store_basic(tmp_path):
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
    st = BM25WhooshStore(index_dir=str(tmp_path))
    docs = [{'doc_id': 'a', 'text': 'apples and bananas'}, {'doc_id': 'b', 'text': 'cars and bikes'}]
    st.index_documents(docs)
    res = st.search('apples', top_k=2)
    assert len(res) == 1 and res[0]['doc_id'] == 'a'

