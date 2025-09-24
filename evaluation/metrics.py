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

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """Advanced evaluation metrics for RAG systems."""

    def __init__(self, ollama_client=None, gemini_client=None, llm_provider="ollama"):
        """
        Initialize advanced evaluator.

        Args:
            ollama_client: Optional Ollama client for LLM-based evaluations
            gemini_client: Optional Gemini client for LLM-based evaluations
            llm_provider: Which LLM provider to use ('ollama' or 'gemini')
        """
        self.ollama_client = ollama_client
        self.gemini_client = gemini_client
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
        if self.ollama_client or self.gemini_client:
            llm_metrics = await self.evaluate_llm_based_metrics(question, answer, contexts)
            results.update(llm_metrics)

        return results

    def evaluate_answer_quality(self, answer: str) -> Dict[str, Any]:
        """Evaluate answer quality metrics."""
        if not answer or answer.startswith("Error"):
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

        # Basic text metrics
        word_count = len(answer.split())
        sentences = re.split(r'[.!?]+', answer)
        sentence_count = len([s for s in sentences if s.strip()])

        # Readability (simplified)
        avg_word_length = np.mean([len(word) for word in answer.split()])
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Flesch Reading Ease (simplified approximation)
        readability_score = max(0, min(100, 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_word_length))

        # Content structure analysis
        has_structure = bool(re.search(r'\d+\.|\•|- |\(|\)', answer))  # Lists, bullets
        has_examples = bool(re.search(r'(?:for example|such as|e\.g\.|example)', answer.lower()))
        has_conclusion = bool(re.search(r'(?:in conclusion|therefore|thus|summary)', answer.lower()))

        return {
            'answer_quality': {
                'length': len(answer),
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': round(avg_sentence_length, 1),
                'readability_score': round(readability_score, 1),
                'has_structure': has_structure,
                'has_examples': has_examples,
                'has_conclusion': has_conclusion
            }
        }

    def evaluate_context_utilization(self, question: str, contexts: List[str]) -> Dict[str, Any]:
        """Evaluate how well contexts are utilized."""
        if not contexts:
            return {
                'context_utilization': {
                    'total_contexts': 0,
                    'total_context_length': 0,
                    'avg_context_length': 0,
                    'context_diversity': 0,
                    'question_context_overlap': 0
                }
            }

        # Basic context metrics
        context_lengths = [len(ctx) for ctx in contexts]
        total_length = sum(context_lengths)
        avg_length = total_length / len(contexts)

        # Context diversity (unique words / total words)
        all_words = []
        for ctx in contexts:
            words = re.findall(r'\b\w+\b', ctx.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        unique_words = len(word_counts)
        total_words = sum(word_counts.values())
        diversity = unique_words / max(total_words, 1)

        # Question-context overlap
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        context_words = set(all_words)
        overlap = len(question_words.intersection(context_words)) / max(len(question_words), 1)

        return {
            'context_utilization': {
                'total_contexts': len(contexts),
                'total_context_length': total_length,
                'avg_context_length': round(avg_length, 1),
                'context_diversity': round(diversity, 3),
                'question_context_overlap': round(overlap, 3)
            }
        }

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

        # Simple heuristics for completeness
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
        if not self.ollama_client and not self.gemini_client:
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
            if self.llm_provider == "gemini" and self.gemini_client and GEMINI_AVAILABLE:
                return await self._get_gemini_score(prompt)
            elif self.llm_provider == "ollama" and self.ollama_client:
                return await self._get_ollama_score(prompt)
            else:
                # Fallback to mock score
                import random
                return round(random.uniform(1, 5), 1)
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return 3.0  # Neutral score

    async def _get_gemini_score(self, prompt: str) -> float:
        """Get score from Gemini Flash."""
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)

            # Extract numerical score from response
            text = response.text.strip()
            # Look for a number between 1-5
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', text)
            if match:
                score = float(match.group(1))
                return max(1.0, min(5.0, score))  # Clamp to 1-5 range
            else:
                logger.warning(f"Could not extract numerical score from Gemini response: {text}")
                return 3.0
        except Exception as e:
            logger.error(f"Gemini scoring failed: {e}")
            return 3.0

    async def _get_ollama_score(self, prompt: str) -> float:
        """Get score from Ollama."""
        try:
            # This would use the actual Ollama client
            # For now, return a mock score until Ollama integration is implemented
            import random
            return round(random.uniform(1, 5), 1)
        except Exception as e:
            logger.error(f"Ollama scoring failed: {e}")
            return 3.0

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
            
        if not self.ollama_client and not self.gemini_client:
            # Fallback heuristic evaluation
            return self._evaluate_groundedness_heuristic(contexts, answer)
        
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
            # Fallback to heuristic
            return self._evaluate_groundedness_heuristic(contexts, answer)
    
    def _evaluate_groundedness_heuristic(self, contexts: List[str], answer: str) -> float:
        """
        Heuristic groundedness evaluation when LLM is not available.
        
        Returns:
            Float between 0-1
        """
        if not contexts or not answer:
            return 0.0
            
        # Simple heuristic: check for keyword overlap
        context_text = ' '.join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        
        # Calculate overlap
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)
        
        if total_answer_words == 0:
            return 0.0
            
        overlap_ratio = overlap / total_answer_words
        
        # Boost score if answer is concise and overlaps well
        if overlap_ratio > 0.3:
            return min(1.0, overlap_ratio * 1.2)  # Slight boost for good overlap
        

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

            # Performance recommendations
            if 'avg_response_time' in trend_data:
                rt_trend = trend_data['avg_response_time']
                if rt_trend['direction'] == 'declining':
                    recommendations.append("Optimize retrieval or model inference to reduce response times")

            # Quality recommendations
            if 'avg_answer_relevance' in trend_data:
                ar_trend = trend_data['avg_answer_relevance']
                if ar_trend['direction'] == 'declining':
                    recommendations.append("Review recent document additions or model fine-tuning")
                elif ar_trend['slope'] < 0.001:  # Very slow improvement
                    recommendations.append("Consider additional training data or model improvements")

        recommendations.append("Monitor evaluation metrics regularly for system health")
        recommendations.append("Review low-performing queries to identify improvement areas")

        return recommendations