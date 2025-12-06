"""
Retrieval strategy for combining semantic and BM25 results.
"""

from typing import Dict, List

from cubo.retrieval.constants import BM25_WEIGHT_DEFAULT, SEMANTIC_WEIGHT_DEFAULT
from cubo.utils.logger import logger


class RetrievalStrategy:
    """Encapsulates the logic for combining semantic and BM25 retrieval results."""

    def __init__(self):
        """Initialize the retrieval strategy."""
        pass

    def combine_results(
        self,
        semantic_candidates: List[Dict],
        bm25_candidates: List[Dict],
        top_k: int,
        semantic_weight: float = SEMANTIC_WEIGHT_DEFAULT,
        bm25_weight: float = BM25_WEIGHT_DEFAULT,
    ) -> List[Dict]:
        """
        Combine semantic and BM25 results using weighted scoring.

        Args:
            semantic_candidates: Results from semantic search
            bm25_candidates: Results from BM25 search
            top_k: Number of final results to return
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores

        Returns:
            Combined and sorted list of candidates
        """
        from cubo.retrieval.fusion import combine_semantic_and_bm25

        try:
            logger.debug(
                f"combine_results semantic={len(semantic_candidates)} bm25={len(bm25_candidates)} top_k={top_k}"
            )
        except Exception:
            pass

        return combine_semantic_and_bm25(
            semantic_candidates,
            bm25_candidates,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            top_k=top_k,
        )

    def apply_postprocessing(
        self,
        candidates: List[Dict],
        top_k: int,
        query: str,
        window_postprocessor=None,
        reranker=None,
        use_reranker: bool = True,
    ) -> List[Dict]:
        """
        Apply window postprocessing and reranking.

        Args:
            candidates: List of candidate documents
            top_k: Number of final results
            query: Original query string
            window_postprocessor: Optional window postprocessor
            reranker: Optional reranker
            use_reranker: Whether to use reranker

        Returns:
            Postprocessed and reranked candidates
        """
        # Apply window postprocessing
        if window_postprocessor:
            try:
                logger.debug(f"postprocess window start count={len(candidates)}")
            except Exception:
                pass
            candidates = window_postprocessor.postprocess_results(candidates)

        # Apply reranking if available and requested
        # Apply reranking if there are enough candidates; allow reranking when
        # candidate count equals requested top_k as some pipelines cap results.
        if use_reranker and reranker and len(candidates) >= top_k:
            try:
                logger.debug(
                    f"rerank start count={len(candidates)} top_k={top_k} use_reranker={use_reranker}"
                )
                reranked = reranker.rerank(query, candidates, max_results=len(candidates))
                if reranked:
                    candidates = reranked
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original order")

        try:
            logger.debug(f"postprocess end count={len(candidates)} (will trim to {top_k})")
        except Exception:
            pass

        return candidates[:top_k]
