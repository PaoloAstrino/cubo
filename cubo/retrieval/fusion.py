"""
Fusion utilities for retrieval: RRF fusion and semantic + BM25 combiner.
"""

from typing import Dict, List

from cubo.config.settings import settings


def rrf_fuse(
    bm25_results: List[Dict],
    faiss_results: List[Dict],
    k: int = None,
    bm25_weight: float = 1.0,
    semantic_weight: float = 1.0,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (Weighted): combine two ranked lists.

    Score = (weight_a / (k + rank_a)) + (weight_b / (k + rank_b))
    """
    if k is None:
        k = settings.retrieval.rrf_k

    fused: Dict[str, Dict] = {}

    # Process BM25
    for i, doc in enumerate(bm25_results):
        rank = i + 1
        doc_id = doc.get("doc_id") or doc.get("id")
        if not doc_id:
            continue

        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "document": doc.get("document", doc.get("text", "")),
                "metadata": doc.get("metadata", {}),
                "similarity": 0.0,
                "bm25_score": doc.get("similarity", 0.0),  # Normalized score
            }

        fused[doc_id]["similarity"] += bm25_weight * (1.0 / (k + rank))

    # Process Semantic
    for i, doc in enumerate(faiss_results):
        rank = i + 1
        doc_id = doc.get("doc_id") or doc.get("id")
        if not doc_id:
            continue

        if doc_id not in fused:
            fused[doc_id] = {
                "id": doc_id,
                "document": doc.get("document", doc.get("text", "")),
                "metadata": doc.get("metadata", {}),
                "similarity": 0.0,
                "semantic_score": doc.get("similarity", 0.0),
            }
        else:
            # Update metadata if missing from BM25 result
            if not fused[doc_id]["document"]:
                fused[doc_id]["document"] = doc.get("document", "")
            fused[doc_id]["semantic_score"] = doc.get("similarity", 0.0)

        fused[doc_id]["similarity"] += semantic_weight * (1.0 / (k + rank))

    # Backward/forward compatibility:
    # - Some call sites expect `similarity` (this module's historical output)
    # - Others expect `score`/`rrf_score` and `doc_id` (strategy.combine_results_rrf)
    for v in fused.values():
        if "doc_id" not in v:
            v["doc_id"] = v.get("id")
        # Use the fused similarity as the canonical RRF score
        if "score" not in v:
            v["score"] = v.get("similarity", 0.0)
        if "rrf_score" not in v:
            v["rrf_score"] = v.get("score", 0.0)

    return list(fused.values())


def combine_semantic_and_bm25(
    semantic_candidates: List[Dict],
    bm25_candidates: List[Dict],
    semantic_weight: float = None,
    bm25_weight: float = None,
    top_k: int = None,
) -> List[Dict]:
    """
    Combine using Weighted RRF (Reciprocal Rank Fusion).
    Replaces naive linear combination with robust rank-based fusion.
    """
    # Load defaults from settings if not provided
    if semantic_weight is None:
        semantic_weight = settings.retrieval.semantic_weight_default
    if bm25_weight is None:
        bm25_weight = settings.retrieval.bm25_weight_default
    if top_k is None:
        top_k = settings.retrieval.default_top_k

    # Use RRF
    fused_results = rrf_fuse(
        bm25_candidates,
        semantic_candidates,
        k=settings.retrieval.rrf_k,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
    )

    # Sort by fused score
    fused_results.sort(key=lambda x: x["similarity"], reverse=True)
    return fused_results[:top_k]
