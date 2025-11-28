"""
RAGAS Evaluator for LightRAG Comparison.

Provides comprehensiveness, diversity, and empowerment metrics using RAGAS library.
Uses GLM model for LLM-based evaluation (configurable).
"""

import asyncio
from typing import Any, Dict, List, Optional

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas import Dataset
    
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RAGASEvaluator:
    """Evaluator for RAGAS metrics (comprehensiveness, diversity, empowerment)."""

    def __init__(self, llm_model: str = "glm-4", embedding_model: Optional[str] = None):
        """
        Initialize RAGAS evaluator.

        Args:
            llm_model: LLM model name for evaluation (default: glm-4 for GLM)
            embedding_model: Optional embedding model override
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not available. Install with: pip install ragas langchain langchain-community"
            )

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        """Get or create LLM instance for evaluation."""
        if self._llm is None:
            try:
                # Try to use Ollama for GLM model
                from langchain_community.llms import Ollama
                
                self._llm = Ollama(model=self.llm_model)
            except Exception as e:
                # Fallback: use OpenAI-compatible endpoint if configured
                try:
                    from langchain_community.chat_models import ChatOpenAI
                    self._llm = ChatOpenAI(model=self.llm_model, temperature=0)
                except Exception:
                    raise RuntimeError(
                        f"Failed to initialize LLM for RAGAS evaluation: {e}"
                    )
        return self._llm

    def _get_embeddings(self):
        """Get or create embeddings instance."""
        if self._embeddings is None:
            if self.embedding_model:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            else:
                # Use default RAGAS embeddings
                self._embeddings = None
        return self._embeddings

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single QA pair using RAGAS metrics.

        Args:
            question: Query string
            answer: Generated answer
            contexts: Retrieved context strings
            ground_truth: Optional ground truth answer

        Returns:
            Dictionary with metric scores
        """
        try:
            # Prepare dataset for RAGAS
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            # Select metrics
            metrics = [
                answer_relevancy,
                faithfulness,
                context_precision if ground_truth else None,
                context_recall if ground_truth else None,
            ]
            metrics = [m for m in metrics if m is not None]

            # Run evaluation
            llm = self._get_llm()
            embeddings = self._get_embeddings()
            
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings if embeddings else None,
            )

            # Extract scores
            scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for metric in metrics:
                    metric_name = metric.name
                    if metric_name in df.columns:
                        scores[metric_name] = float(df[metric_name].iloc[0])
            
            # Map to LightRAG comparison dimensions
            # Comprehensiveness ≈ context_recall + faithfulness
            # Diversity ≈ answer_relevancy (diverse info coverage)
            # Empowerment ≈ context_precision (useful info)
            
            comprehensiveness = scores.get("context_recall", 0) * 0.6 + scores.get("faithfulness", 0) * 0.4
            diversity = scores.get("answer_relevancy", 0)
            empowerment = scores.get("context_precision", 0)
            
            overall = (comprehensiveness + diversity + empowerment) / 3.0

            return {
                "comprehensiveness": comprehensiveness,
                "diversity": diversity,
                "empowerment": empowerment,
                "overall": overall,
                # Raw RAGAS scores
                "ragas_raw": scores,
            }

        except Exception as e:
            # Return zeros on error
            return {
                "comprehensiveness": 0.0,
                "diversity": 0.0,
                "empowerment": 0.0,
                "overall": 0.0,
                "error": str(e),
            }

    async def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple QA pairs in batch.

        Args:
            questions: List of queries
            answers: List of generated answers
            contexts_list: List of context lists
            ground_truths: Optional list of ground truth answers

        Returns:
            List of metric dictionaries
        """
        results = []
        for i, (q, a, c) in enumerate(zip(questions, answers, contexts_list)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            result = await self.evaluate_single(q, a, c, ground_truth=gt)
            results.append(result)
        return results


def get_ragas_evaluator(
    llm_model: str = "glm-4", embedding_model: Optional[str] = None
) -> Optional[RAGASEvaluator]:
    """
    Get RAGAS evaluator instance.

    Args:
        llm_model: LLM model name (default: glm-4)
        embedding_model: Optional embedding model override

    Returns:
        RAGASEvaluator instance or None if RAGAS not available
    """
    if not RAGAS_AVAILABLE:
        return None
    
    try:
        return RAGASEvaluator(llm_model=llm_model, embedding_model=embedding_model)
    except Exception:
        return None
