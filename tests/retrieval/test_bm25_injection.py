from cubo.retrieval.bm25_sqlite_store import BM25SqliteStore


def test_bm25_query_sanitization(tmp_path):
    index_dir = tmp_path
    store = BM25SqliteStore(str(index_dir))

    # Ensure DB/tables are initialized by the store
    store.index_documents([{"doc_id": "d1", "text": "hello world", "metadata": {}}])

    # Malicious queries (should be sanitized and not cause SQL errors)
    malicious = '"; DROP TABLE documents; --'
    results = store.search(malicious, top_k=5)
    # No exception and results is a list
    assert isinstance(results, list)

    # Complex injection-like payload
    payload = "foo OR 1=1; --"
    results2 = store.search(payload, top_k=5)
    assert isinstance(results2, list)
