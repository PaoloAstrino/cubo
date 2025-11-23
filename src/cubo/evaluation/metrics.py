"""
Advanced RAG Evaluation Metrics
Implements comprehensive evaluation metrics beyond the basic triad.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import re
from collections import Counter
import time

# google.generativeai/Gemini integration removed to avoid external API calls
GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """Advanced evaluation metrics for RAG systems."""

    def __init__(self, ollama_client=None, llm_provider="ollama"):
        """
        Initialize advanced evaluator.

        Args:
            ollama_client: Optional Ollama client for LLM-based evaluations
            gemini_client: removed - Gemini integrations are disabled.
            llm_provider: Which LLM provider to use ('ollama')
        """
        self.ollama_client = ollama_client
        self.llm_provider = llm_provider

    async def evaluate_comprehensive(self, question: str, answer: str,
                                     contexts: List[str], response_time: float) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite.

        Returns detailed metrics beyond the basic triad.
        """
        results = {}

        # Basic RAG Triad (would be computed separately)
        # results.update(await self.evaluate_rag_triad(question, answer, contexts))

        # Advanced Metrics
        results.update(self.evaluate_answer_quality(answer))
        results.update(self.evaluate_context_utilization(question, contexts))
        results.update(self.evaluate_response_efficiency(answer, response_time))
        results.update(self.evaluate_information_completeness(question, answer, contexts))

        # Groundedness evaluation
        groundedness_score = await self.evaluate_groundedness(contexts, answer)
        results['groundedness_score'] = groundedness_score

        # LLM-based advanced metrics (if available)
        if self.ollama_client:
            llm_metrics = await self.evaluate_llm_based_metrics(question, answer, contexts)
            results.update(llm_metrics)

        return results

    def evaluate_answer_quality(self, answer: str) -> Dict[str, Any]:
        """Evaluate answer quality metrics."""
        if not answer or answer.startswith("Error"):
            return self._create_empty_answer_quality_result()

        # Calculate basic text metrics
        basic_metrics = self._calculate_basic_text_metrics(answer)

        # Calculate readability score
        readability_score = self._calculate_readability_score(answer, basic_metrics)

        # Analyze content structure
        structure_analysis = self._analyze_content_structure(answer)

        return {
            'answer_quality': {
                'length': len(answer),
                'word_count': basic_metrics['word_count'],
                'sentence_count': basic_metrics['sentence_count'],
                'avg_sentence_length': basic_metrics['avg_sentence_length'],
                'readability_score': readability_score,
                'has_structure': structure_analysis['has_structure'],
                'has_examples': structure_analysis['has_examples'],
                'has_conclusion': structure_analysis['has_conclusion']
            }
        }

    def _create_empty_answer_quality_result(self) -> Dict[str, Any]:
        """Create empty answer quality result for invalid inputs."""
        return {
            'answer_quality': {
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'readability_score': 0,
                'has_structure': False,
                'has_examples': False,
                'has_conclusion': False
            }
        }

    def _calculate_basic_text_metrics(self, answer: str) -> Dict[str, Any]:
        """Calculate basic text metrics like word count, sentence count, etc."""
        word_count = len(answer.split())
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)

        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length
        }

    def _calculate_readability_score(self, answer: str, basic_metrics: Dict[str, Any]) -> float:
        """Calculate Flesch Reading Ease score."""
        avg_word_length = np.mean([len(word) for word in answer.split()])
        avg_sentence_length = basic_metrics['avg_sentence_length']

        # Flesch Reading Ease (simplified approximation)
        return max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length))

    def _analyze_content_structure(self, answer: str) -> Dict[str, bool]:
        """Analyze content structure for lists, examples, and conclusions."""
        has_structure = bool(re.search(r'\d+\.|\•|- |\(|\)', answer))  # Lists, bullets
        has_examples = bool(re.search(r'(?:for example|such as|e\.g\.|example)', answer.lower()))
        has_conclusion = bool(re.search(r'(?:in conclusion|therefore|thus|summary)', answer.lower()))

        return {
            'has_structure': has_structure,
            'has_examples': has_examples,
            'has_conclusion': has_conclusion
        }

    async def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question.

        Answer relevance measures whether the answer actually addresses the question asked.

        Returns:
            Float between 0-1, where 1.0 means perfectly relevant, or None if evaluation fails
        """
        if not question or not answer:
            return None

        if not self.ollama_client:
            # No LLM available for evaluation
            return None

        try:
            relevance_prompt = f"""
            Evaluate how well this answer addresses the question.

            Question: {question}

            Answer: {answer}

            Rate the answer relevance on a scale of 1-5:
            1 = Answer does not address the question at all
            2 = Answer partially addresses the question but misses key points
            3 = Answer addresses the main question but could be more complete
            4 = Answer fully addresses the question with good detail
            5 = Answer perfectly addresses the question comprehensively

            Consider:
            - Does the answer directly answer what was asked?
            - Is the answer complete and accurate?
            - Does the answer stay on topic?

            Rate only with a number (1-5):
            """

            score = await self._get_llm_score(relevance_prompt)

            # Convert 1-5 scale to 0-1 scale
            return (score - 1) / 4.0

        except Exception as e:
            logger.error(f"LLM answer relevance evaluation failed: {e}")
            return None

    async def evaluate_context_relevance(self, question: str, contexts: List[str]) -> float:
        """
        Evaluate how relevant the retrieved contexts are to the question.

        Context relevance measures whether the retrieved information is useful for answering the question.

        Returns:
            Float between 0-1, where 1.0 means perfectly relevant contexts, or None if evaluation fails
        """
        if not question or not contexts:
            return None

        if not self.ollama_client:
            # No LLM available for evaluation
            return None

        try:
            # Combine contexts for evaluation
            context_text = ' '.join(contexts[:3])  # Use first 3 contexts to avoid token limits

            relevance_prompt = f"""
            Evaluate how relevant these retrieved contexts are to the question.

            Question: {question}

            Retrieved Contexts: {context_text}

            Rate the context relevance on a scale of 1-5:
            1 = Contexts are completely irrelevant to the question
            2 = Contexts have some tangential relevance but miss the main topic
            3 = Contexts are somewhat relevant but could be better
            4 = Contexts are highly relevant and useful for answering
            5 = Contexts are perfectly relevant and contain exactly the needed information

            Consider:
            - Do the contexts contain information that helps answer the question?
            - Are the contexts on the right topic?
            - Is the information in contexts sufficient to answer the question?

            Rate only with a number (1-5):
            """

            score = await self._get_llm_score(relevance_prompt)

            # Convert 1-5 scale to 0-1 scale
            return (score - 1) / 4.0

        except Exception as e:
            logger.error(f"LLM context relevance evaluation failed: {e}")
            return None








    def _calculate_length_appropriateness_score(self, answer: str) -> float:
        """Calculate score based on answer length appropriateness."""
        answer_length = len(answer.split())
        if 10 <= answer_length <= 200:
            return 0.3
        elif answer_length < 10:
            return 0.1
        else:
            return 0.2

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question (why, how, what, or other)."""
        question_lower = question.lower()
        if question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('how'):
            return 'how'
        elif question_lower.startswith('what'):
            return 'what'
        else:
            return 'other'

    def _calculate_question_type_relevance(self, question_type: str, answer: str) -> float:
        """Calculate relevance score based on question type and answer structure."""
        answer_lower = answer.lower()

        if question_type == 'why':
            # For "why" questions, look for explanations
            return self._check_why_question_relevance(answer_lower)
        elif question_type == 'how':
            # For "how" questions, look for process descriptions
            return self._check_how_question_relevance(answer_lower)
        elif question_type == 'what':
            # For "what" questions, look for definitions or descriptions
            return self._check_what_question_relevance(answer)

        return 0.0

    def _check_why_question_relevance(self, answer_lower: str) -> float:
        """Check if answer addresses a 'why' question with explanations."""
        if any(word in answer_lower for word in ['because', 'due to', 'since', 'as', 'so']):
            return 0.3
        return 0.0



    def evaluate_context_utilization(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        """Evaluate how well contexts are utilized."""
        if not contexts:
            return self._create_empty_context_utilization_result()

        # Calculate basic context metrics
        basic_metrics = self._calculate_basic_context_metrics(contexts)

        # Calculate context diversity
        diversity = self._calculate_context_diversity(contexts)

        # Calculate question-context overlap
        overlap = self._calculate_question_context_overlap(question, contexts)

        return {
            'context_utilization': {
                'total_contexts': len(contexts),
                'total_context_length': basic_metrics['total_length'],
                'avg_context_length': round(basic_metrics['avg_length'], 1),
                'context_diversity': round(diversity, 3),
                'question_context_overlap': round(overlap, 3)
            }
        }

    def _create_empty_context_utilization_result(self) -> Dict[str, Any]:
        """Create empty context utilization result for invalid inputs."""
        return {
            'context_utilization': {
                'total_contexts': 0,
                'total_context_length': 0,
                'avg_context_length': 0,
                'context_diversity': 0,
                'question_context_overlap': 0
            }
        }

    def _calculate_basic_context_metrics(self, contexts: List[str]) -> Dict[str, Any]:
        """Calculate basic context metrics like lengths and counts."""
        context_lengths = [len(ctx) for ctx in contexts]
        total_length = sum(context_lengths)
        avg_length = total_length / len(contexts)

        return {
            'total_length': total_length,
            'avg_length': avg_length
        }

    def _calculate_context_diversity(self, contexts: List[str]) -> float:
        """Calculate context diversity based on unique words."""
        all_words = []
        for ctx in contexts:
            words = re.findall(r'\b\w+\b', ctx.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        unique_words = len(word_counts)
        total_words = sum(word_counts.values())

        return unique_words / max(total_words, 1)

    def _calculate_question_context_overlap(self, question: str, contexts: List[str]) -> float:
        """Calculate overlap between question and context words."""
        # Get all words from contexts
        all_context_words = []
        for ctx in contexts:
            words = re.findall(r'\b\w+\b', ctx.lower())
            all_context_words.extend(words)

        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        context_words = set(all_context_words)

        overlap = len(question_words.intersection(context_words)) / max(len(question_words), 1)
        return overlap

    def evaluate_response_efficiency(self, answer: str, response_time: float) -> Dict[str, Any]:
        """Evaluate response efficiency metrics."""
        if not answer or answer.startswith("Error"):
            return {
                'response_efficiency': {
                    'response_time': response_time,
                    'words_per_second': 0,
                    'chars_per_second': 0,
                    'efficiency_score': 0
                }
            }

        word_count = len(answer.split())
        char_count = len(answer)

        words_per_second = word_count / max(response_time, 0.1)
        chars_per_second = char_count / max(response_time, 0.1)

        # Efficiency score (words per second, normalized)
        efficiency_score = min(1.0, words_per_second / 50.0)  # 50 WPS as "perfect"

        return {
            'response_efficiency': {
                'response_time': round(response_time, 3),
                'words_per_second': round(words_per_second, 2),
                'chars_per_second': round(chars_per_second, 2),
                'efficiency_score': round(efficiency_score, 3)
            }
        }

    def evaluate_information_completeness(self, question: str, answer: str,
                                          contexts: List[str]) -> Dict[str, Any]:
        """Evaluate information completeness."""
        if not answer or not contexts:
            return {
                'information_completeness': {
                    'answer_context_coverage': 0,
                    'unique_information_ratio': 0,
                    'question_answer_alignment': 0
                }
            }

        # Basic calculations for completeness
        answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
        context_words = set()
        for ctx in contexts:
            context_words.update(re.findall(r'\b\w+\b', ctx.lower()))

        # Coverage: how much of answer comes from contexts
        coverage = len(answer_words.intersection(context_words)) / max(len(answer_words), 1)

        # Unique information: how much new info is in answer vs contexts
        unique_in_answer = len(answer_words - context_words) / max(len(answer_words), 1)

        # Question-answer alignment (simplified)
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        alignment = len(question_words.intersection(answer_words)) / max(len(question_words), 1)

        return {
            'information_completeness': {
                'answer_context_coverage': round(coverage, 3),
                'unique_information_ratio': round(unique_in_answer, 3),
                'question_answer_alignment': round(alignment, 3)
            }
        }

    async def evaluate_llm_based_metrics(self, question: str, answer: str,
                                        contexts: List[str]) -> Dict[str, Any]:
        """Advanced LLM-based evaluation metrics."""
        if not self.ollama_client:
            return {}

        results = {}

        try:
            # Answer coherence
            coherence_prompt = f"""
            Rate the coherence and logical flow of this answer on a scale of 1-5.
            1 = Disorganized and hard to follow
            5 = Perfectly coherent and logical

            Question: {question}
            Answer: {answer}

            Rate only with a number:"""

            coherence_score = await self._get_llm_score(coherence_prompt)

            # Answer specificity
            specificity_prompt = f"""
            Rate how specific and detailed this answer is on a scale of 1-5.
            1 = Very vague and general
            5 = Highly specific with concrete details

            Question: {question}
            Answer: {answer}

            Rate only with a number:"""

            specificity_score = await self._get_llm_score(specificity_prompt)

            # Factual accuracy check
            accuracy_prompt = f"""
            Based on the provided contexts, rate the factual accuracy of this answer on a scale of 1-5.
            1 = Contains factual errors
            5 = Completely factually accurate

            Contexts: {' '.join(contexts[:2])}...  # First 2 contexts
            Answer: {answer}

            Rate only with a number:"""

            accuracy_score = await self._get_llm_score(accuracy_prompt)

            results['llm_metrics'] = {
                'coherence_score': coherence_score,
                'specificity_score': specificity_score,
                'factual_accuracy_score': accuracy_score
            }

        except Exception as e:
            logger.error(f"LLM-based evaluation failed: {e}")
            results['llm_metrics'] = {
                'error': str(e)
            }

        return results

    async def _get_llm_score(self, prompt: str) -> float:
        """Get numerical score from LLM."""
        try:
            if self.llm_provider == "ollama" and self.ollama_client:
                score = await self._get_ollama_score(prompt)
                if score is None:
                    return None
                return score
            else:
                # No LLM available
                return None
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return None  # Neutral score

    # Gemini scoring removed (external API). Only local scoring via _get_ollama_score supported.

    async def _get_ollama_score(self, prompt: str) -> float:
        """Get score from Ollama."""
        try:
            import ollama
            import asyncio

            # Get model name from config or use default
            try:
                from src.cubo.config import config
                model_name = config.get("selected_llm_model") or config.get("llm_model") or "llama3.2"
            except:
                model_name = "llama3.2"  # Default fallback

            # Create scoring prompt
            scoring_messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Rate the given item on a scale of 1-5 and respond ONLY with a single number (1, 2, 3, 4, or 5). Do not include any other text or explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Run Ollama scoring with timeout
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: ollama.chat(model=model_name, messages=scoring_messages)),
                timeout=30.0  # 30 second timeout
            )

            # Extract numerical score from response
            text = response['message']['content'].strip()
            logger.debug(f"Ollama response for scoring: {text}")

            # Look for a number between 1-5
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                logger.debug(f"Extracted Ollama score: {score}")
                return max(1.0, min(5.0, score))  # Clamp to 1-5 range
            else:
                logger.warning(f"Could not extract numerical score from Ollama response: {text}")
                return None

        except asyncio.TimeoutError:
            logger.error("Ollama scoring timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama scoring failed: {e}")
            return None

    async def evaluate_groundedness(self, contexts: List[str], answer: str) -> float:
        """
        Evaluate how well the answer is grounded in the provided contexts.

        Groundedness measures whether the answer is supported by the contexts
        and doesn't contain unsupported claims (hallucinations).

        Returns:
            Float between 0-1, where 1.0 means fully grounded
        """
        if not contexts or not answer:
            return 0.0

        if not self.ollama_client:
            # No LLM available for evaluation
            return None

        try:
            # Combine contexts for evaluation
            context_text = ' '.join(contexts[:3])  # Use first 3 contexts to avoid token limits

            groundedness_prompt = f"""
            Evaluate how well this answer is grounded in the provided contexts.

            Contexts: {context_text}

            Answer: {answer}

            Rate the groundedness on a scale of 1-5:
            1 = Answer contains significant unsupported claims or hallucinations
            2 = Answer has some unsupported information but mostly grounded
            3 = Answer is reasonably grounded with minor gaps
            4 = Answer is well-grounded with good support from contexts
            5 = Answer is perfectly grounded and fully supported by contexts

            Consider:
            - Does the answer make claims not supported by contexts?
            - Are key facts and information backed by the provided contexts?
            - Does the answer stay within the bounds of the given information?

            Rate only with a number (1-5):
            """

            score = await self._get_llm_score(groundedness_prompt)

            # Convert 1-5 scale to 0-1 scale
            return (score - 1) / 4.0

        except Exception as e:
            logger.error(f"LLM groundedness evaluation failed: {e}")
            return None

class PerformanceAnalyzer:
    """Analyze performance trends and patterns in evaluation data."""

    def __init__(self, db):
        """
        Initialize performance analyzer.

        Args:
            db: EvaluationDatabase instance
        """
        self.db = db

    def analyze_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze performance trends over the specified number of days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with trend analysis
        """
        try:
            # Get metrics summary for the period
            summary = self.db.get_metrics_summary(days)

            # Get trend data
            trends = self.db.get_trends(days)

            # Combine into comprehensive analysis
            analysis = {
                'summary': summary,
                'trends': trends,
                'recommendations': self._generate_recommendations(trends)
            }

            return analysis

        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
            return {
                'error': str(e),
                'summary': {},
                'trends': {},
                'recommendations': []
            }

    def _generate_recommendations(self, trends: Dict) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []

        if 'trends' in trends:
            trend_data = trends['trends']

            # Generate performance-based recommendations
            recommendations.extend(self._generate_performance_recommendations(trend_data))

            # Generate quality-based recommendations
            recommendations.extend(self._generate_quality_recommendations(trend_data))

        # Add general recommendations
        recommendations.extend(self._generate_general_recommendations())

        return recommendations

    def _generate_performance_recommendations(self, trend_data: Dict) -> List[str]:
        """Generate recommendations based on performance trends."""
        recommendations = []

        if 'avg_response_time' in trend_data:
            rt_trend = trend_data['avg_response_time']
            if rt_trend.get('direction') == 'declining':
                recommendations.append("Optimize retrieval or model inference to reduce response times")

        return recommendations

    def _generate_quality_recommendations(self, trend_data: Dict) -> List[str]:
        """Generate recommendations based on quality trends."""
        recommendations = []

        if 'avg_answer_relevance' in trend_data:
            ar_trend = trend_data['avg_answer_relevance']
            if ar_trend.get('direction') == 'declining':
                recommendations.append("Review recent document additions or model fine-tuning")
            elif ar_trend.get('slope', 0) < 0.001:  # Very slow improvement
                recommendations.append("Consider additional training data or model improvements")

        return recommendations

    def _generate_general_recommendations(self) -> List[str]:
        """Generate general monitoring recommendations."""
        return [
            "Monitor evaluation metrics regularly for system health",
            "Review low-performing queries to identify improvement areas"
        ]


class IRMetricsEvaluator:
    """Information Retrieval metrics for evaluating retrieval quality."""

    @staticmethod
    def compute_recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Compute Recall@K metric.
        
        Recall@K measures the proportion of relevant documents retrieved in top K results.
        
        Args:
            relevant_ids: List of ground-truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in ranked order)
            k: Cutoff rank
            
        Returns:
            Recall@K score (0.0 to 1.0)
            
        Example:
            >>> relevant = ['doc1', 'doc2', 'doc3']
            >>> retrieved = ['doc1', 'doc4', 'doc2']
            >>> compute_recall_at_k(relevant, retrieved, k=3)
            0.6667  # 2 out of 3 relevant docs found
        """
        if not relevant_ids:
            return 0.0
        
        # Take only top K retrieved results
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        # Count how many relevant docs are in top K
        relevant_retrieved = len(relevant_set.intersection(retrieved_at_k))
        
        return relevant_retrieved / len(relevant_set)
    
    @staticmethod
    def compute_precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
        """
        Compute Precision@K metric.
        
        Precision@K measures the proportion of retrieved documents that are relevant.
        
        Args:
            relevant_ids: List of ground-truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in ranked order)
            k: Cutoff rank
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not retrieved_ids or k == 0:
            return 0.0
        
        retrieved_at_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    @staticmethod
    def compute_ndcg_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int, 
                          relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K (nDCG@K).
        
        nDCG@K measures ranking quality with position-dependent discount.
        
        Args:
            relevant_ids: List of ground-truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in ranked order)
            k: Cutoff rank
            relevance_scores: Optional dict mapping doc_id -> relevance score (default: binary 1/0)
            
        Returns:
            nDCG@K score (0.0 to 1.0)
            
        Formula:
            DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
            IDCG@K = DCG of perfect ranking
            nDCG@K = DCG@K / IDCG@K
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0
        
        # Default to binary relevance (1 for relevant, 0 for non-relevant)
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
        
        # Compute DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        
        # Compute Ideal DCG (IDCG) - best possible ranking
        ideal_ranking = sorted(relevance_scores.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_ranking[:k], start=1):
            idcg += rel / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR is the reciprocal of the rank of the first relevant document.
        
        Args:
            relevant_ids: List of ground-truth relevant document IDs
            retrieved_ids: List of retrieved document IDs (in ranked order)
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def evaluate_retrieval(
        question_id: str,
        retrieved_ids: List[str],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 20],
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality with multiple IR metrics.
        
        Args:
            question_id: Question identifier for ground truth lookup
            retrieved_ids: List of retrieved document IDs (in ranked order)
            ground_truth: Dict mapping question_id -> list of relevant doc IDs
            k_values: List of K values to compute metrics for
            relevance_scores: Optional dict mapping doc_id -> relevance score
            
        Returns:
            Dictionary with IR metrics (recall, precision, nDCG, MRR)
            
        Example:
            >>> ground_truth = {'q1': ['doc1', 'doc2', 'doc3']}
            >>> retrieved = ['doc1', 'doc4', 'doc2', 'doc5', 'doc3']
            >>> evaluate_retrieval('q1', retrieved, ground_truth, k_values=[5, 10])
            {
                'recall_at_k': {5: 1.0, 10: 1.0},
                'precision_at_k': {5: 0.6, 10: 0.3},
                'ndcg_at_k': {5: 0.89, 10: 0.89},
                'mrr': 1.0
            }
        """
        relevant_ids = ground_truth.get(question_id, [])
        
        if not relevant_ids:
            logger.warning(f"No ground truth found for question_id: {question_id}")
            return {
                'recall_at_k': {k: 0.0 for k in k_values},
                'precision_at_k': {k: 0.0 for k in k_values},
                'ndcg_at_k': {k: 0.0 for k in k_values},
                'mrr': 0.0,
                'error': 'no_ground_truth'
            }
        
        results = {
            'recall_at_k': {},
            'precision_at_k': {},
            'ndcg_at_k': {},
            'mrr': IRMetricsEvaluator.compute_mrr(relevant_ids, retrieved_ids)
        }
        
        for k in k_values:
            results['recall_at_k'][k] = IRMetricsEvaluator.compute_recall_at_k(
                relevant_ids, retrieved_ids, k
            )
            results['precision_at_k'][k] = IRMetricsEvaluator.compute_precision_at_k(
                relevant_ids, retrieved_ids, k
            )
            results['ndcg_at_k'][k] = IRMetricsEvaluator.compute_ndcg_at_k(
                relevant_ids, retrieved_ids, k, relevance_scores
            )
        
        return results


class GroundTruthLoader:
    """Load and manage ground truth data for retrieval evaluation."""
    
    @staticmethod
    def load_beir_format(filepath: str) -> Dict[str, List[str]]:
        """
        Load ground truth in BeIR format (qrels.tsv or JSON).
        
        BeIR format: query-id \t corpus-id \t score
        
        Args:
            filepath: Path to qrels file
            
        Returns:
            Dictionary mapping query_id -> list of relevant doc IDs
        """
        import json
        from pathlib import Path
        
        ground_truth = {}
        path = Path(filepath)
        
        if path.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Expecting format: {query_id: {doc_id: relevance_score}}
            for query_id, doc_scores in data.items():
                relevant_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
                ground_truth[query_id] = relevant_docs
                
        elif path.suffix in ['.tsv', '.txt']:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        query_id, doc_id, score = parts[0], parts[1], float(parts[2])
                        if score > 0:  # Only relevant docs
                            if query_id not in ground_truth:
                                ground_truth[query_id] = []
                            ground_truth[query_id].append(doc_id)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return ground_truth
    
    @staticmethod
    def load_custom_format(filepath: str) -> Dict[str, List[str]]:
        """
        Load ground truth in custom JSON format.
        
        Format: {question_id: [list of relevant doc IDs]}
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary mapping question_id -> list of relevant doc IDs
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        return ground_truth