"""
Evaluation metrics for CUBO.
Provides IR metrics (Recall@K, Precision@K, nDCG@K, MRR) and evaluation utilities.
"""

import math
from typing import Any, Dict, List, Optional

# Re-export from benchmarks
from benchmarks.utils.metrics import AdvancedEvaluator, GroundTruthLoader


class IRMetricsEvaluator:
    """Evaluator for Information Retrieval (IR) metrics."""

    @staticmethod
    def compute_recall_at_k(
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            relevant_ids: List of ground truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Number of top results to consider
            
        Returns:
            Recall score between 0.0 and 1.0
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        retrieved_k = set(retrieved_ids[:k])
        hits = len(relevant_set & retrieved_k)
        return hits / len(relevant_set)

    @staticmethod
    def compute_precision_at_k(
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            relevant_ids: List of ground truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Number of top results to consider
            
        Returns:
            Precision score between 0.0 and 1.0
        """
        if k == 0:
            return 0.0
        
        relevant_set = set(relevant_ids)
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
        return hits / k

    @staticmethod
    def compute_ndcg_at_k(
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute nDCG@K (Normalized Discounted Cumulative Gain).
        
        Args:
            relevant_ids: List of ground truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            k: Number of top results to consider
            relevance_scores: Optional dict mapping doc_id to relevance score
            
        Returns:
            nDCG score between 0.0 and 1.0
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                rel = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
                dcg += rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
        
        # Calculate IDCG (ideal DCG)
        if relevance_scores:
            # Sort relevant docs by relevance score
            sorted_rels = sorted(
                [relevance_scores.get(doc_id, 1.0) for doc_id in relevant_ids],
                reverse=True
            )[:k]
        else:
            sorted_rels = [1.0] * min(len(relevant_ids), k)
        
        idcg = 0.0
        for i, rel in enumerate(sorted_rels):
            idcg += rel / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

    @staticmethod
    def compute_mrr(
        relevant_ids: List[str],
        retrieved_ids: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_ids: List of ground truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in order)
            
        Returns:
            MRR score between 0.0 and 1.0
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0

    @staticmethod
    def evaluate_retrieval(
        question_id: str,
        retrieved_ids: List[str],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance for a single query.
        
        Args:
            question_id: Query identifier
            retrieved_ids: List of retrieved document IDs
            ground_truth: Dict mapping question_id to list of relevant doc IDs
            k_values: List of K values for metrics computation
            
        Returns:
            Dict with recall_at_k, precision_at_k, ndcg_at_k, and mrr metrics
        """
        if question_id not in ground_truth:
            return {
                "error": "no_ground_truth",
                "recall_at_k": {k: 0.0 for k in k_values},
                "precision_at_k": {k: 0.0 for k in k_values},
                "ndcg_at_k": {k: 0.0 for k in k_values},
                "mrr": 0.0,
            }
        
        relevant_ids = ground_truth[question_id]
        
        recall_at_k = {}
        precision_at_k = {}
        ndcg_at_k = {}
        
        for k in k_values:
            recall_at_k[k] = IRMetricsEvaluator.compute_recall_at_k(
                relevant_ids, retrieved_ids, k
            )
            precision_at_k[k] = IRMetricsEvaluator.compute_precision_at_k(
                relevant_ids, retrieved_ids, k
            )
            ndcg_at_k[k] = IRMetricsEvaluator.compute_ndcg_at_k(
                relevant_ids, retrieved_ids, k
            )
        
        mrr = IRMetricsEvaluator.compute_mrr(relevant_ids, retrieved_ids)
        
        return {
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "ndcg_at_k": ndcg_at_k,
            "mrr": mrr,
        }


__all__ = [
    "IRMetricsEvaluator",
    "AdvancedEvaluator", 
    "GroundTruthLoader",
]
