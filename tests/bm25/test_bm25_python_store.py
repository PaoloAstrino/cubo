from src.cubo.retrieval.bm25_python_store import BM25PythonStore


def test_python_store_search():
    # Direct Python store test (not using the default backend)
    st = BM25PythonStore()
    docs = [
        {"doc_id": "a", "text": "I love apples and bananas."},
        {"doc_id": "b", "text": "I drive a car."},
    ]
    st.index_documents(docs)
    res = st.search("apples", top_k=2)
    assert len(res) == 1 and res[0]["doc_id"] == "a"


def test_python_store_compute_score():
    # Direct Python store test (not using the default backend)
    st = BM25PythonStore()
    docs = [
        {"doc_id": "a", "text": "apples bananas apples apples"},
        {"doc_id": "b", "text": "apples"},
    ]
    st.index_documents(docs)
    score_a = st.compute_score(["apples"], "a")
    score_b = st.compute_score(["apples"], "b")
    assert score_a >= score_b
