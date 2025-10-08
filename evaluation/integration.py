"""
CUBO Evaluation Integration
Connects the evaluation system with the main CUBO application.
"""

import asyncio
import logging
import time
import json
import datetime
from typing import Dict, Any, Optional, List
import uuid
from dataclasses import asdict
from pathlib import Path

from evaluation.database import EvaluationDatabase, QueryEvaluation
from evaluation.metrics import AdvancedEvaluator

logger = logging.getLogger(__name__)

class EvaluationIntegrator:
    """Integrates evaluation system with CUBO components."""

    def __init__(self, generator=None, retriever=None):
        """
        Initialize evaluation integrator.

        Args:
            generator: ResponseGenerator instance
            retriever: DocumentRetriever instance
        """
        self.db = EvaluationDatabase()
        self.generator = generator
        self.retriever = retriever
        self.session_id = str(uuid.uuid4())[:8]

        # Initialize evaluator with LLM client based on config
        self.evaluator = self._init_evaluator()

    def _init_evaluator(self) -> AdvancedEvaluator:
        """Initialize evaluator with appropriate LLM client."""
        try:
            config = self._load_evaluation_config()
            llm_provider = config.get('llm_provider', 'ollama')

            if llm_provider == 'gemini':
                return self._init_gemini_evaluator(config)
            else:
                return self._init_basic_evaluator()

        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            return AdvancedEvaluator(llm_provider='ollama')

    def _load_evaluation_config(self) -> dict:
        """Load evaluation configuration from config.json."""
        import json
        config_path = Path(__file__).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('evaluation', {})

    def _init_gemini_evaluator(self, eval_config: dict) -> AdvancedEvaluator:
        """Initialize Gemini-based evaluator."""
        api_key = eval_config.get('gemini_api_key')
        if not api_key or api_key == 'your_gemini_api_key_here':
            logger.warning("Gemini API key not configured, falling back to basic evaluation")
            return self._init_basic_evaluator()

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            logger.info("Gemini client initialized for evaluation")
            return AdvancedEvaluator(gemini_client=genai, llm_provider='gemini')
        except ImportError:
            logger.warning("google-generativeai not installed, falling back to basic evaluation")
            return self._init_basic_evaluator()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return self._init_basic_evaluator()

    def _init_basic_evaluator(self) -> AdvancedEvaluator:
        """Initialize basic evaluator without LLM client."""
        logger.info("Using basic evaluation (no LLM client)")
        return AdvancedEvaluator(llm_provider='ollama')

    def set_components(self, generator, retriever):
        """Set or update the generator and retriever components."""
        self.generator = generator
        self.retriever = retriever

    async def evaluate_query(self, question: str, answer: str,
                             contexts: List[str], response_time: float,
                             model_used: str = "llama3.2:latest",
                             error_occurred: bool = False,
                             error_message: Optional[str] = None) -> QueryEvaluation:
        """
        Evaluate a complete query and store results.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context chunks
            response_time: Time taken to generate response (seconds)
            model_used: Which model was used
            error_occurred: Whether an error occurred
            error_message: Error message if applicable

        Returns:
            QueryEvaluation object with all metrics
        """
        import datetime

        # Basic evaluation data
        evaluation = QueryEvaluation(
            timestamp=datetime.datetime.now().isoformat(),
            session_id=self.session_id,
            question=question,
            answer=answer,
            response_time=response_time,
            contexts=contexts,
            context_metadata=[],  # Would be populated from retriever
            model_used=model_used,
            embedding_model="embeddinggemma-300m",  # From config
            retrieval_method="sentence_window",
            chunking_method="sentence_window",
            answer_relevance_score=0.0,  # Will be computed
            context_relevance_score=0.0,  # Will be computed
            groundedness_score=0.0,  # Will be computed
            answer_length=len(answer) if answer else 0,
            context_count=len(contexts),
            total_context_length=sum(len(ctx) for ctx in contexts),
            average_context_similarity=0.0,  # Would be computed from retriever
            answer_confidence=0.0,  # Not available yet
            has_answer=bool(answer and not answer.startswith("Error")),
            is_fallback_response=answer and "unable to generate" in answer.lower(),
            error_occurred=error_occurred,
            error_message=error_message,
            user_rating=None,  # Not available yet
            user_feedback=None,  # Not available yet
            llm_metrics=None  # Will be set if LLM evaluation succeeds
        )

        # Track evaluation completion
        evaluation_completed = False

        # Compute advanced metrics
        if not error_occurred:
            try:
                # Run comprehensive evaluation
                advanced_metrics = await self.evaluator.evaluate_comprehensive(
                    question, answer, contexts, response_time
                )

                # Extract RAG triad scores using LLM evaluation
                logger.debug(f"Evaluating query: {question[:50]}...")
                answer_relevance = await self.evaluator.evaluate_answer_relevance(question, answer)
                context_relevance = await self.evaluator.evaluate_context_relevance(question, contexts)
                groundedness = await self.evaluator.evaluate_groundedness(contexts, answer)

                # Only proceed if we have valid LLM scores for all three core metrics
                if (answer_relevance is not None and
                        context_relevance is not None and
                        groundedness is not None):
                    evaluation.answer_relevance_score = answer_relevance
                    evaluation.context_relevance_score = context_relevance
                    evaluation.groundedness_score = groundedness
                    logger.debug(f"Answer relevance score: {evaluation.answer_relevance_score}")
                    logger.debug(f"Context relevance score: {evaluation.context_relevance_score}")
                    logger.debug(f"Groundedness score: {evaluation.groundedness_score}")

                    # Extract LLM metrics if available
                    llm_metrics = advanced_metrics.get('llm_metrics')
                    if llm_metrics:
                        evaluation.llm_metrics = llm_metrics

                    # Store advanced metrics in metadata
                    evaluation.context_metadata = [{
                        'metric_type': 'advanced_evaluation',
                        'data': advanced_metrics
                    }]

                    # Mark as successfully evaluated
                    evaluation_completed = True
                else:
                    logger.warning(f"LLM evaluation failed for query: {question[:50]}... - AR={answer_relevance}, CR={context_relevance}, G={groundedness}")
                    # Don't store this evaluation, leave it for retry
                    return None

            except Exception as e:
                logger.error(f"Advanced evaluation failed: {e}")
                error_occurred = True

        # Store in database only if evaluation completed successfully
        if evaluation_completed and not error_occurred:
            self.db.store_evaluation(evaluation)
            logger.info(f"Query evaluation stored: {question[:50]}... | Scores: AR={evaluation.answer_relevance_score:.2f}, CR={evaluation.context_relevance_score:.2f}, G={evaluation.groundedness_score:.2f}")
            return evaluation
        else:
            if not evaluation_completed:
                logger.warning(f"Skipping storage of failed LLM evaluation for: {question[:50]}...")
            else:
                logger.warning(f"Skipping storage due to error for: {question[:50]}...")
            return None

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """Simple heuristic for answer relevance."""
        if not answer or answer.startswith("Error"):
            return 0.0

        # Simple keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        overlap = len(question_words.intersection(answer_words))
        coverage = overlap / max(len(question_words), 1)

        return min(coverage * 1.5, 1.0)  # Boost slightly, cap at 1.0

    def _compute_context_relevance(self, question: str, contexts: List[str]) -> float:
        """Simple heuristic for context relevance."""
        if not contexts:
            return 0.0

        question_words = set(question.lower().split())
        total_overlap = 0

        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(question_words.intersection(context_words))
            total_overlap += overlap

        avg_overlap = total_overlap / len(contexts)
        coverage = avg_overlap / max(len(question_words), 1)

        return min(coverage * 2.0, 1.0)  # Boost more, cap at 1.0

    def _compute_groundedness(self, contexts: List[str], answer: str) -> float:
        """Simple heuristic for groundedness."""
        if not contexts or not answer or answer.startswith("Error"):
            return 0.0

        # Check if answer contains phrases from contexts
        context_text = ' '.join(contexts).lower()
        answer_lower = answer.lower()

        # Simple word overlap check
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())

        overlap = len(answer_words.intersection(context_words))
        coverage = overlap / max(len(answer_words), 1)

        return min(coverage, 1.0)

    def get_recent_evaluations(self, limit: int = 10) -> List[QueryEvaluation]:
        """Get recent evaluations."""
        return self.db.get_recent_evaluations(limit)

    def get_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get evaluation metrics summary."""
        return self.db.get_metrics_summary(days)

    def export_evaluation_data(self, output_path: str, days: int = 30, format_type: str = "json"):
        """Export evaluation data."""
        if format_type.lower() == "csv":
            self.db.export_to_csv(output_path, days)
        else:
            # JSON export
            evaluations = self.db.get_evaluations_by_date_range(
                (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat(),
                datetime.datetime.now().isoformat()
            )

            data = {
                'export_info': {
                    'generated_at': datetime.datetime.now().isoformat(),
                    'days': days,
                    'total_records': len(evaluations)
                },
                'evaluations': [asdict(eval) for eval in evaluations]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

# Global evaluation integrator instance
_evaluation_integrator = None

def get_evaluation_integrator(generator=None, retriever=None) -> EvaluationIntegrator:
    """Get or create the global evaluation integrator instance."""
    global _evaluation_integrator

    if _evaluation_integrator is None:
        _evaluation_integrator = EvaluationIntegrator(generator, retriever)
    else:
        # Update components if provided
        if generator or retriever:
            _evaluation_integrator.set_components(generator, retriever)

    return _evaluation_integrator

async def evaluate_query_async(question: str, answer: str, contexts: List[str],
                               response_time: float, **kwargs) -> Optional[QueryEvaluation]:
    """
    Convenience function to evaluate a query asynchronously.

    Returns the evaluation result or None if evaluation fails.
    """
    try:
        integrator = get_evaluation_integrator()
        return await integrator.evaluate_query(question, answer, contexts, response_time, **kwargs)
    except Exception as e:
        logger.error(f"Query evaluation failed: {e}")
        return None

def evaluate_query_sync(question: str, answer: str, contexts: List[str],
                      response_time: float, **kwargs) -> Optional[QueryEvaluation]:
    """
    Convenience function to evaluate a query synchronously.

    Returns the evaluation result or None if evaluation fails.
    """
    try:
        integrator = get_evaluation_integrator()

        # Run async evaluation in new event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            integrator.evaluate_query(question, answer, contexts, response_time, **kwargs)
        )

        loop.close()
        return result

    except Exception as e:
        logger.error(f"Query evaluation failed: {e}")
        return None


def save_query_data_sync(question: str, answer: str, contexts: List[str],
                       response_time: float, model_used: str = "llama3.2:latest",
                       error_occurred: bool = False, error_message: Optional[str] = None) -> bool:
    """
    Save query data without running evaluation.

    This stores basic query information in the database without computing metrics.
    Evaluation can be run later manually.

    Returns True if data was saved successfully, False otherwise.
    """
    try:
        integrator = get_evaluation_integrator()

        # Create evaluation record
        evaluation = _create_basic_evaluation_record(
            question, answer, contexts, response_time, model_used,
            error_occurred, error_message, integrator.session_id
        )

        # Store in database
        _store_evaluation_record(evaluation, integrator.db, question)

        return True

    except Exception as e:
        logger.error(f"Failed to save query data: {e}")
        return False


def _create_basic_evaluation_record(question: str, answer: str, contexts: List[str],
                                   response_time: float, model_used: str,
                                   error_occurred: bool, error_message: Optional[str],
                                   session_id: str) -> QueryEvaluation:
    """Create a basic evaluation record without computed metrics."""
    import datetime

    return QueryEvaluation(
        timestamp=datetime.datetime.now().isoformat(),
        session_id=session_id,
        question=question,
        answer=answer,
        response_time=response_time,
        contexts=contexts,
        context_metadata=[],
        model_used=model_used,
        embedding_model="embeddinggemma-300m",
        retrieval_method="sentence_window",
        chunking_method="sentence_window",
        answer_relevance_score=None,  # Not computed yet
        context_relevance_score=None,  # Not computed yet
        groundedness_score=None,  # Not computed yet
        answer_length=len(answer) if answer else 0,
        context_count=len(contexts),
        total_context_length=sum(len(ctx) for ctx in contexts),
        average_context_similarity=0.0,
        answer_confidence=0.0,
        has_answer=bool(answer and not answer.startswith("Error")),
        is_fallback_response=answer and "unable to generate" in answer.lower(),
        error_occurred=error_occurred,
        error_message=error_message,
        user_rating=None,
        user_feedback=None,
        llm_metrics=None  # Will be set when evaluation runs
    )


def _store_evaluation_record(evaluation: QueryEvaluation, db: EvaluationDatabase, question: str) -> None:
    """Store the evaluation record in the database."""
    db.store_evaluation(evaluation)
    logger.info(f"Query data saved (no evaluation): {question[:50]}...")