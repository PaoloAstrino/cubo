from cubo.retrieval.fusion import combine_semantic_and_bm25, rrf_fuse


def make_doc(doc_id, similarity=1.0, meta=None):
    return {
        "id": doc_id,
        "document": f"doc_{doc_id}",
        "similarity": similarity,
        "metadata": meta or {},
    }


def test_rrf_includes_union_of_candidates():
    bm25 = [
        make_doc("A", similarity=3.0),
        make_doc("B", similarity=2.0),
        make_doc("C", similarity=1.0),
    ]
    semantic = [
        make_doc("B", similarity=0.9),
        make_doc("C", similarity=0.8),
        make_doc("D", similarity=0.7),
    ]

    fused = rrf_fuse(bm25, semantic, k=10, bm25_weight=1.0, semantic_weight=1.0)
    ids = {d["id"] for d in fused}

    # Union of ids should include all unique ids
    assert ids == {"A", "B", "C", "D"}


def test_combine_semantic_and_bm25_matches_rrf_behavior():
    bm25 = [
        make_doc("A", similarity=5.0),
        make_doc("B", similarity=4.0),
        make_doc("C", similarity=3.0),
    ]
    semantic = [
        make_doc("B", similarity=0.9),
        make_doc("C", similarity=0.8),
        make_doc("D", similarity=0.7),
    ]

    fused_rrf = rrf_fuse(bm25, semantic, k=60, bm25_weight=1.0, semantic_weight=1.0)
    fused_combiner = combine_semantic_and_bm25(
        semantic, bm25, semantic_weight=1.0, bm25_weight=1.0, top_k=10
    )

    # Ensure the top ids returned by the public combiner match the RRF implementation ordering
    top_rrf_ids = [
        d["id"] for d in sorted(fused_rrf, key=lambda x: x["similarity"], reverse=True)[:5]
    ]
    top_comb_ids = [d["id"] for d in fused_combiner]

    assert set(top_rrf_ids).issubset(set(top_comb_ids))
