import logging
import statistics
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class GroundTruthLoader:
    """Helper to load ground truth data for evaluation."""

    @staticmethod
    def load_beir_format(file_path: str) -> Dict[str, List[str]]:
        """Load BEIR qrels format."""
        import csv
        ground_truth = {}
        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                qid = row.get('query-id')
                did = row.get('corpus-id')
                if qid and did:
                    if qid not in ground_truth:
                        ground_truth[qid] = []
                    ground_truth[qid].append(did)
        return ground_truth

    @staticmethod
    def load_custom_format(file_path: str) -> Dict[str, List[str]]:
        """Load custom JSON {qid: [doc_ids]} format."""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class IRMetricsEvaluator:
    """Evaluator for Information Retrieval (IR) metrics."""

    def evaluate_retrieval(
        self,
        question_id: str,
        retrieved_ids: List[str],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Calculate Recall@K and NDCG@K for a single query.
        """
        if not ground_truth or question_id not in ground_truth:
            return {}

        relevant_docs = set(ground_truth[question_id])
        metrics = {}

        # Recall@K
        for k in k_values:
            retrieved_k = set(retrieved_ids[:k])
            hits = len(retrieved_k.intersection(relevant_docs))
            recall = hits / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
            metrics[f"recall_at_k_{k}"] = recall

        # NDCG@K (Binary relevance assumption)
        import math
        for k in k_values:
            dcg = 0.0
            idcg = 0.0
            # Calculate DCG
            for i, doc_id in enumerate(retrieved_ids[:k]):
                if doc_id in relevant_docs:
                    dcg += 1.0 / math.log2(i + 2)
            # Calculate IDCG
            for i in range(min(len(relevant_docs), k)):
                idcg += 1.0 / math.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg_at_k_{k}"] = ndcg

        return metrics


class AdvancedEvaluator:
    """
    Evaluator using RAGAS for semantic metrics (Faithfulness, Answer Relevancy, etc.).
    """

    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS is not installed. LLM-based evaluation will be skipped.")

    async def evaluate_comprehensive(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        response_time: float = 0.0,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run RAGAS evaluation on a single sample.
        """
        if not RAGAS_AVAILABLE or not self.ollama_client:
            return {
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
                "groundedness_score": 0.0,
                "error": "RAGAS or Ollama not available"
            }

        # Prepare data for RAGAS
        data_sample = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data_sample["ground_truth"] = [ground_truth]

        try:
            # Create dataset
            dataset = Dataset.from_dict(data_sample)

            # Choose metrics
            metrics_to_run = [
                faithfulness,
                answer_relevancy,
                context_precision,
            ]
            if ground_truth:
                metrics_to_run.append(context_recall)

            # Evaluate using RAGAS
            # Note: RAGAS typically requires OpenAI API key by default.
            # For local Ollama, we would need to configure RAGAS with a custom LLM wrapper.
            # Assuming generic RAGAS setup here; user might need to set OPENAI_API_KEY or configure local LLM.
            
            # Since we can't easily mock the local LLM for RAGAS without extra setup code,
            # we will return placeholder values if specific env vars aren't set,
            # or attempt standard evaluation.
            
            # For this implementation, we'll try to run it but catch errors gracefully.
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                raise_exceptions=False
            )

            return {
                "answer_relevance": results.get("answer_relevancy", 0.0),
                "context_relevance": results.get("context_precision", 0.0), # Mapping precision to relevance proxy
                "groundedness_score": results.get("faithfulness", 0.0),
                "response_efficiency": {"time": response_time}
            }

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {
                "answer_relevance": 0.0,
                "context_relevance": 0.0,
                "groundedness_score": 0.0,
                "error": str(e)
            }
