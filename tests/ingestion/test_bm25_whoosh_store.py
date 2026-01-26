def test_bm25_python_store_basic(tmp_path):
    from cubo.retrieval.bm25_python_store import BM25PythonStore

    st = BM25PythonStore()
    docs = [
        {"doc_id": "a", "text": "apples and bananas"},
        {"doc_id": "b", "text": "cars and bikes"},
    ]
    st.index_documents(docs)
    _res = st.search("apples", top_k=2)
    assert len(_res) == 1 and _res[0]["doc_id"] == "a"
