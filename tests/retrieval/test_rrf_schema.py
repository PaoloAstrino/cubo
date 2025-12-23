from cubo.retrieval.fusion import rrf_fuse


def make_doc(doc_id, similarity=1.0, meta=None):
    return {
        "id": doc_id,
        "document": f"doc_{doc_id}",
        "similarity": similarity,
        "metadata": meta or {},
    }


def test_rrf_fuse_emits_doc_id_and_score_fields():
    bm25 = [make_doc("A", similarity=3.0)]
    semantic = [make_doc("B", similarity=0.9)]

    fused = rrf_fuse(bm25, semantic, k=10, bm25_weight=1.0, semantic_weight=1.0)

    assert isinstance(fused, list) and len(fused) >= 1
    for entry in fused:
        # Ensure backwards-compatible id field exists
        assert entry.get("id") is not None
        # New compatibility fields must be present
        assert entry.get("doc_id") is not None
        assert "score" in entry
        assert "rrf_score" in entry
        # score must be numeric
        assert isinstance(entry.get("score"), (int, float))
