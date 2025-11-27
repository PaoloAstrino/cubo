"""
Retrieval Orchestrator - Coordinates retrieval operations.

This module provides a clean orchestration layer that coordinates between
specialized retrieval components (semantic search, BM25, tiered retrieval, etc.)
without being a God Object itself.

The orchestrator follows the Facade pattern - it provides a unified interface
while delegating to specialized components.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.cubo.retrieval.constants import (
    BM25_NORMALIZATION_FACTOR,
    BM25_WEIGHT_DEFAULT,
    BM25_WEIGHT_DETAILED,
    INITIAL_RETRIEVAL_MULTIPLIER,
    KEYWORD_BOOST_FACTOR,
    MIN_BM25_THRESHOLD,
    SEMANTIC_WEIGHT_DEFAULT,
    SEMANTIC_WEIGHT_DETAILED,
)
from src.cubo.utils.logger import logger


class TieredRetrievalManager:
    """
    Manages the three-tier retrieval strategy:

    Tier 1: Summary prefilter (fast, broad)
    Tier 2: Scaffold compression (semantic grouping)
    Tier 3: Dense vectors + BM25 hybrid (precise)
    """

    def __init__(
        self,
        summary_embeddings: Optional[np.ndarray] = None,
        summary_chunk_ids: Optional[List[str]] = None,
        scaffold_retriever: Optional[Any] = None,
        summary_weight: float = 0.2,
        scaffold_weight: float = 0.3,
        dense_weight: float = 0.5,
    ):
        self.summary_embeddings = summary_embeddings
        self.summary_chunk_ids = summary_chunk_ids or []
        self.scaffold_retriever = scaffold_retriever
        self.summary_weight = summary_weight
        self.scaffold_weight = scaffold_weight
        self.dense_weight = dense_weight

    @property
    def has_summary_prefilter(self) -> bool:
        """Check if summary prefilter is available."""
        return self.summary_embeddings is not None and len(self.summary_embeddings) > 0

    @property
    def has_scaffold_retrieval(self) -> bool:
        """Check if scaffold retrieval is available."""
        return self.scaffold_retriever is not None

    def retrieve_via_summary(self, query_embedding: np.ndarray, k: int) -> Set[str]:
        """
        Tier 1: Use summary embeddings to quickly identify candidate chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of summary matches to return

        Returns:
            Set of chunk IDs that match the query via summary
        """
        if not self.has_summary_prefilter:
            return set()

        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Handle dimension mismatch
        if query_embedding.shape[1] != self.summary_embeddings.shape[1]:
            logger.debug(
                f"Summary embedding dimension mismatch: "
                f"query={query_embedding.shape[1]}, summary={self.summary_embeddings.shape[1]}"
            )
            return set()

        # Compute cosine similarity
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
        summary_norm = self.summary_embeddings / (
            np.linalg.norm(self.summary_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = np.dot(query_norm, summary_norm.T).flatten()

        # Get top-k indices
        top_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return chunk IDs
        return {self.summary_chunk_ids[i] for i in top_indices if i < len(self.summary_chunk_ids)}

    def retrieve_via_scaffold(self, query: str, top_k: int) -> tuple[Set[str], Dict[str, float]]:
        """
        Tier 2: Use scaffold compression for semantic grouping.

        Args:
            query: Query string
            top_k: Number of results to expand

        Returns:
            Tuple of (chunk_ids set, scores dict)
        """
        if not self.has_scaffold_retrieval:
            return set(), {}

        chunk_ids: Set[str] = set()
        scores: Dict[str, float] = {}

        try:
            scaffold_results = self.scaffold_retriever.retrieve_scaffolds(
                query=query, top_k=max(5, top_k // 2), expand_to_chunks=True
            )
            for result in scaffold_results:
                result_chunk_ids = result.get("chunk_ids", [])
                score = result.get("score", 0.5)
                for cid in result_chunk_ids:
                    chunk_ids.add(cid)
                    scores[cid] = max(scores.get(cid, 0), score)
        except Exception as e:
            logger.warning(f"Scaffold retrieval failed: {e}")

        return chunk_ids, scores

    def apply_tiered_boosting(
        self,
        candidates: List[Dict],
        summary_chunk_ids: Set[str],
        scaffold_chunk_ids: Set[str],
        scaffold_scores: Dict[str, float],
        chunk_id_extractor: callable,
    ) -> List[Dict]:
        """
        Apply score boosting based on tiered retrieval matches.

        Args:
            candidates: Base candidates from dense+BM25 retrieval
            summary_chunk_ids: Chunk IDs from summary prefilter
            scaffold_chunk_ids: Chunk IDs from scaffold expansion
            scaffold_scores: Scaffold match scores by chunk ID
            chunk_id_extractor: Function to extract chunk ID from candidate

        Returns:
            Candidates with boosted scores
        """
        if not summary_chunk_ids and not scaffold_chunk_ids:
            return candidates

        boosted = []
        for candidate in candidates:
            chunk_id = chunk_id_extractor(candidate)
            base_score = candidate.get("similarity", 0.5)
            boost = 0.0

            # Boost for summary prefilter match
            if chunk_id and chunk_id in summary_chunk_ids:
                boost += self.summary_weight * 0.2

            # Boost for scaffold match
            if chunk_id and chunk_id in scaffold_chunk_ids:
                scaffold_score = scaffold_scores.get(chunk_id, 0.5)
                boost += self.scaffold_weight * scaffold_score * 0.3

            boosted_candidate = candidate.copy()
            boosted_candidate["similarity"] = min(1.0, base_score + boost)
            if boost > 0:
                boosted_candidate["tier_boost"] = boost
            boosted.append(boosted_candidate)

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return boosted


class HybridScorer:
    """
    Handles hybrid scoring between semantic and BM25 results.

    Provides methods for combining scores from different retrieval methods
    with configurable weights.
    """

    def __init__(
        self,
        bm25_normalization_factor: float = BM25_NORMALIZATION_FACTOR,
        default_semantic_weight: float = SEMANTIC_WEIGHT_DEFAULT,
        default_bm25_weight: float = BM25_WEIGHT_DEFAULT,
    ):
        self.bm25_normalization_factor = bm25_normalization_factor
        self.default_semantic_weight = default_semantic_weight
        self.default_bm25_weight = default_bm25_weight

    def normalize_bm25_score(self, raw_score: float) -> float:
        """Normalize BM25 score to [0, 1] range."""
        return min(raw_score / self.bm25_normalization_factor, 1.0)

    def apply_keyword_boost(
        self,
        base_similarity: float,
        bm25_score: float,
        min_threshold: float = MIN_BM25_THRESHOLD,
        boost_factor: float = KEYWORD_BOOST_FACTOR,
    ) -> float:
        """
        Apply keyword boost to base similarity score.

        Args:
            base_similarity: Semantic similarity score
            bm25_score: Raw BM25 score
            min_threshold: Minimum normalized BM25 to apply boost
            boost_factor: Boost contribution factor

        Returns:
            Boosted combined score
        """
        normalized_bm25 = self.normalize_bm25_score(bm25_score)

        if normalized_bm25 > min_threshold:
            boost = boost_factor * normalized_bm25
            combined_score = min(base_similarity + boost, 1.0)
            logger.debug(
                f"BM25 BOOST: raw={bm25_score:.3f}, norm={normalized_bm25:.3f}, "
                f"semantic={base_similarity:.3f}, boost={boost:.3f}, combined={combined_score:.3f}"
            )
            return combined_score

        return base_similarity

    def apply_keyword_boost_detailed(
        self,
        base_similarity: float,
        bm25_score: float,
        semantic_weight: float = SEMANTIC_WEIGHT_DETAILED,
        bm25_weight: float = BM25_WEIGHT_DETAILED,
    ) -> Dict[str, float]:
        """
        Apply keyword boost with detailed score breakdown.

        Args:
            base_similarity: Semantic similarity score
            bm25_score: Raw BM25 score
            semantic_weight: Weight for semantic contribution
            bm25_weight: Weight for BM25 contribution

        Returns:
            Dictionary with detailed score breakdown
        """
        normalized_bm25 = self.normalize_bm25_score(bm25_score)
        semantic_contribution = semantic_weight * base_similarity
        bm25_contribution = bm25_weight * normalized_bm25
        final_score = min(semantic_contribution + bm25_contribution, 1.0)

        logger.debug(
            f"DETAILED SCORE: semantic={base_similarity:.3f} (contrib={semantic_contribution:.3f}), "
            f"bm25_raw={bm25_score:.3f}, bm25_norm={normalized_bm25:.3f} "
            f"(contrib={bm25_contribution:.3f}), final={final_score:.3f}"
        )

        return {
            "final_score": final_score,
            "semantic_score": base_similarity,
            "bm25_score": bm25_score,
            "bm25_normalized": normalized_bm25,
            "semantic_contribution": semantic_contribution,
            "bm25_contribution": bm25_contribution,
        }

    def combine_results(
        self,
        semantic_candidates: List[Dict],
        bm25_candidates: List[Dict],
        top_k: int,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Combine semantic and BM25 candidate lists.

        Args:
            semantic_candidates: Results from semantic search
            bm25_candidates: Results from BM25 search
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (default from config)
            bm25_weight: Weight for BM25 scores (default from config)

        Returns:
            Combined and sorted list of candidates
        """
        semantic_weight = semantic_weight or self.default_semantic_weight
        bm25_weight = bm25_weight or self.default_bm25_weight

        # Build combined map
        combined: Dict[str, Dict] = {}

        for cand in semantic_candidates:
            doc_key = cand.get("document", "")[:100]
            if doc_key not in combined:
                combined[doc_key] = {
                    "id": cand.get("id"),
                    "document": cand.get("document", ""),
                    "metadata": cand.get("metadata", {}),
                    "semantic_score": cand.get("similarity", 0.0),
                    "bm25_score": 0.0,
                }
            else:
                combined[doc_key]["semantic_score"] = max(
                    combined[doc_key]["semantic_score"], cand.get("similarity", 0.0)
                )

        for cand in bm25_candidates:
            doc_key = cand.get("document", "")[:100]
            if doc_key not in combined:
                combined[doc_key] = {
                    "id": cand.get("id"),
                    "document": cand.get("document", ""),
                    "metadata": cand.get("metadata", {}),
                    "semantic_score": 0.0,
                    "bm25_score": cand.get("similarity", 0.0),
                }
            else:
                combined[doc_key]["bm25_score"] = max(
                    combined[doc_key]["bm25_score"], cand.get("similarity", 0.0)
                )

        # Calculate combined scores
        results = []
        for doc_data in combined.values():
            combined_score = (
                semantic_weight * doc_data["semantic_score"] + bm25_weight * doc_data["bm25_score"]
            )
            results.append(
                {
                    "id": doc_data.get("id"),
                    "document": doc_data["document"],
                    "metadata": doc_data.get("metadata", {}),
                    "similarity": combined_score,
                    "base_similarity": doc_data["semantic_score"],
                    "bm25_normalized": doc_data["bm25_score"],
                }
            )

        # Sort and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


class DeduplicationManager:
    """
    Manages deduplication of retrieval results.

    Supports both content-based deduplication and cluster-based deduplication
    using pre-computed deduplication maps.
    """

    def __init__(
        self,
        enabled: bool = False,
        cluster_lookup: Optional[Dict[str, int]] = None,
        representatives: Optional[Dict[int, Dict[str, Any]]] = None,
        canonical_lookup: Optional[Dict[str, str]] = None,
    ):
        self.enabled = enabled
        self.cluster_lookup = cluster_lookup or {}
        self.representatives = representatives or {}
        self.canonical_lookup = canonical_lookup or {}
        self._map_loaded = bool(cluster_lookup)

    @property
    def is_ready(self) -> bool:
        """Check if deduplication is enabled and configured."""
        return self.enabled and self._map_loaded

    def deduplicate(
        self,
        results: List[Dict],
        chunk_id_extractor: callable,
    ) -> List[Dict]:
        """
        Deduplicate retrieval results.

        Args:
            results: List of result dictionaries
            chunk_id_extractor: Function to extract chunk ID from result

        Returns:
            Deduplicated results
        """
        if not results:
            return []

        if not self.is_ready:
            return self._dedup_by_content(results)

        return self._dedup_by_cluster(results, chunk_id_extractor)

    def _dedup_by_content(self, results: List[Dict]) -> List[Dict]:
        """Simple content-based deduplication."""
        seen_content: Set[str] = set()
        unique_results: List[Dict] = []

        for result in results:
            content = result.get("document", result.get("content", ""))
            if content in seen_content:
                continue
            seen_content.add(content)
            unique_results.append(result)

        return unique_results

    def _dedup_by_cluster(
        self,
        results: List[Dict],
        chunk_id_extractor: callable,
    ) -> List[Dict]:
        """Cluster-based deduplication using pre-computed maps."""
        seen: Set[Any] = set()
        deduped: List[Dict] = []

        for result in results:
            chunk_id = chunk_id_extractor(result)
            cluster_id = self.cluster_lookup.get(chunk_id)
            key: Any = cluster_id if cluster_id is not None else chunk_id or result.get("document")

            if key in seen:
                continue
            seen.add(key)

            # Promote representative if available
            if cluster_id is not None:
                representative = self.representatives.get(cluster_id)
                if representative and chunk_id and representative["chunk_id"] != chunk_id:
                    result = self._promote_representative(result, representative, cluster_id)

            deduped.append(result)

        return deduped

    def _promote_representative(
        self,
        result: Dict,
        rep: Dict[str, Any],
        cluster_id: int,
    ) -> Dict:
        """Promote cluster representative in result metadata."""
        updated = result.copy()
        metadata = dict(result.get("metadata") or {})
        metadata["dedup_cluster_id"] = cluster_id
        metadata["canonical_chunk_id"] = rep.get("chunk_id")
        updated["metadata"] = metadata
        updated["canonical_chunk_id"] = rep.get("chunk_id")
        return updated


class RetrievalOrchestrator:
    """
    Coordinates retrieval operations across specialized components.

    This is the main entry point for retrieval operations, providing
    a clean interface while delegating to specialized managers.

    Components:
    - TieredRetrievalManager: Handles multi-tier retrieval
    - HybridScorer: Combines semantic and BM25 scores
    - DeduplicationManager: Removes duplicate results
    """

    def __init__(
        self,
        tiered_manager: Optional[TieredRetrievalManager] = None,
        hybrid_scorer: Optional[HybridScorer] = None,
        dedup_manager: Optional[DeduplicationManager] = None,
    ):
        self.tiered_manager = tiered_manager or TieredRetrievalManager()
        self.hybrid_scorer = hybrid_scorer or HybridScorer()
        self.dedup_manager = dedup_manager or DeduplicationManager()

    def calculate_initial_top_k(self, top_k: int, use_sentence_window: bool) -> int:
        """
        Calculate initial number of candidates to retrieve before reranking.

        Args:
            top_k: Final number of results desired
            use_sentence_window: Whether sentence window is enabled

        Returns:
            Number of initial candidates to retrieve
        """
        return top_k * INITIAL_RETRIEVAL_MULTIPLIER if use_sentence_window else top_k

    def execute_tiered_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        summary_prefilter_k: int = 20,
        scaffold_top_k: int = 5,
    ) -> tuple[Set[str], Set[str], Dict[str, float]]:
        """
        Execute tiered retrieval (summary + scaffold).

        Args:
            query: Query string
            query_embedding: Query embedding vector
            summary_prefilter_k: Number of summary matches
            scaffold_top_k: Number of scaffold expansions

        Returns:
            Tuple of (summary_chunk_ids, scaffold_chunk_ids, scaffold_scores)
        """
        # Tier 1: Summary prefilter
        summary_chunk_ids = set()
        if self.tiered_manager.has_summary_prefilter:
            try:
                summary_chunk_ids = self.tiered_manager.retrieve_via_summary(
                    query_embedding, summary_prefilter_k
                )
                logger.debug(f"Summary prefilter returned {len(summary_chunk_ids)} chunk IDs")
            except Exception as e:
                logger.warning(f"Summary prefilter failed: {e}")

        # Tier 2: Scaffold compression
        scaffold_chunk_ids = set()
        scaffold_scores = {}
        if self.tiered_manager.has_scaffold_retrieval:
            try:
                scaffold_chunk_ids, scaffold_scores = self.tiered_manager.retrieve_via_scaffold(
                    query, scaffold_top_k
                )
                logger.debug(f"Scaffold retrieval returned {len(scaffold_chunk_ids)} chunk IDs")
            except Exception as e:
                logger.warning(f"Scaffold retrieval failed: {e}")

        return summary_chunk_ids, scaffold_chunk_ids, scaffold_scores

    def combine_and_boost(
        self,
        semantic_candidates: List[Dict],
        bm25_candidates: List[Dict],
        summary_chunk_ids: Set[str],
        scaffold_chunk_ids: Set[str],
        scaffold_scores: Dict[str, float],
        top_k: int,
        chunk_id_extractor: callable,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Combine retrieval results and apply tiered boosting.

        Args:
            semantic_candidates: Semantic search results
            bm25_candidates: BM25 search results
            summary_chunk_ids: IDs from summary prefilter
            scaffold_chunk_ids: IDs from scaffold expansion
            scaffold_scores: Scaffold match scores
            top_k: Number of final results
            chunk_id_extractor: Function to extract chunk ID
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores

        Returns:
            Combined, boosted, and sorted candidates
        """
        # Combine semantic and BM25
        combined = self.hybrid_scorer.combine_results(
            semantic_candidates,
            bm25_candidates,
            top_k * 2,  # Get more for tiered boosting
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        # Apply tiered boosting
        boosted = self.tiered_manager.apply_tiered_boosting(
            combined,
            summary_chunk_ids,
            scaffold_chunk_ids,
            scaffold_scores,
            chunk_id_extractor,
        )

        return boosted[:top_k]

    def deduplicate_results(
        self,
        results: List[Dict],
        chunk_id_extractor: callable,
    ) -> List[Dict]:
        """
        Deduplicate retrieval results.

        Args:
            results: List of result dictionaries
            chunk_id_extractor: Function to extract chunk ID

        Returns:
            Deduplicated results
        """
        return self.dedup_manager.deduplicate(results, chunk_id_extractor)
