import pytest


def test_bm25_python_store_basic(tmp_path):
    from src.cubo.retrieval.bm25_python_store import BM25PythonStore

    st = BM25PythonStore()
    docs = [
        {"doc_id": "a", "text": "apples and bananas"},
        {"doc_id": "b", "text": "cars and bikes"},
    ]
    st.index_documents(docs)
    res = st.search("apples", top_k=2)
    assert len(res) == 1 and res[0]["doc_id"] == "a"
