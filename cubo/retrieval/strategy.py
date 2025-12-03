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
            candidates = window_postprocessor.postprocess_results(candidates)

        # Apply reranking if available and requested
        # Apply reranking if there are enough candidates; allow reranking when
        # candidate count equals requested top_k as some pipelines cap results.
        if use_reranker and reranker and len(candidates) >= top_k:
            try:
                reranked = reranker.rerank(query, candidates, max_results=len(candidates))
                if reranked:
                    candidates = reranked
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original order")

        return candidates[:top_k]
