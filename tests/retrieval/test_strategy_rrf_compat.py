from cubo.retrieval.strategy import RetrievalStrategy


def test_combine_results_rrf_handles_various_fused_shapes(monkeypatch):
    # Prepare simple candidate lookup
    semantic = [{"id": "X", "document": "x", "similarity": 0.9}]
    bm25 = [{"id": "Y", "document": "y", "similarity": 0.8}]

    # Case A: fused entries use 'id' and 'similarity' (old shape)
    def fake_rrf_old(bm, sem, k=60, bm25_weight=1.0, semantic_weight=1.0):
        return [{"id": "X", "similarity": 0.5}, {"id": "Y", "similarity": 0.4}]

    monkeypatch.setattr("cubo.retrieval.fusion.rrf_fuse", fake_rrf_old)

    strat = RetrievalStrategy()
    res_old = strat.combine_results_rrf(semantic, bm25, top_k=2)
    assert [r.get("doc_id") or r.get("id") for r in res_old] == ["X", "Y"]

    # Case B: fused entries use 'doc_id' and 'score' (new shape)
    def fake_rrf_new(bm, sem, k=60, bm25_weight=1.0, semantic_weight=1.0):
        return [{"doc_id": "Y", "score": 0.6}, {"doc_id": "X", "score": 0.5}]

    monkeypatch.setattr("cubo.retrieval.fusion.rrf_fuse", fake_rrf_new)
    res_new = strat.combine_results_rrf(semantic, bm25, top_k=2)
    assert [r.get("doc_id") or r.get("id") for r in res_new] == ["Y", "X"]
