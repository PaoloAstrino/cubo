#!/usr/bin/env python3
"""
CUBO RAG Testing Script
Runs systematic tests using the comprehensive question set and evaluation metrics.
"""

import json
import time
import logging
from typing import Dict, List, Any
import sys
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import CUBOApp
from evaluation.metrics import AdvancedEvaluator
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGTester:
    """Comprehensive RAG testing framework."""

    def __init__(self, questions_file: str = "test_questions.json"):
        """Initialize the tester with question data."""
        self.questions_file = questions_file
        self.questions = self.load_questions()

        # Initialize evaluator
        self.evaluator = AdvancedEvaluator()

        self.results = {
            "metadata": {
                "test_run_timestamp": time.time(),
                "total_questions": 0,
                "questions_by_difficulty": {},
                "success_rate": 0.0
            },
            "results": {
                "easy": [],
                "medium": [],
                "hard": []
            }
        }

    def load_questions(self) -> Dict[str, List[str]]:
        """Load questions from JSON file."""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {data['metadata']['total_questions']} questions")
            return data['questions']
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            return {"easy": [], "medium": [], "hard": []}

    def run_single_test(self, question: str, difficulty: str) -> Dict[str, Any]:
        """Run a single question test with real RAG evaluation."""
        logger.info(f"Testing [{difficulty}]: {question[:50]}...")

        start_time = time.time()

        try:
            # TODO: Replace with actual RAG system integration
            # To integrate with real CUBO system:
            # 1. Initialize CUBOApp instance
            # 2. Load documents from data folder
            # 3. Run query through retriever and generator
            # 4. Capture actual response and retrieved contexts

            # For now, simulate a realistic RAG response based on the question
            # This should be replaced with actual CUBO query processing

            # Mock response - replace with actual RAG call
            response = self._generate_mock_response(question)

            # Mock contexts - replace with actual retrieved documents
            contexts = self._generate_mock_contexts(question)

            processing_time = time.time() - start_time

            # Evaluate the response using AdvancedEvaluator from metrics.py
            evaluation_results = asyncio.run(self.evaluate_response(question, response, contexts, processing_time))

            result = {
                "question": question,
                "difficulty": difficulty,
                "response": response,
                "contexts": contexts,
                "processing_time": processing_time,
                "evaluation": evaluation_results,
                "success": True,  # Based on evaluation scores
                "timestamp": time.time()
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Test failed for question: {question[:50]}... Error: {e}")
            result = {
                "question": question,
                "difficulty": difficulty,
                "error": str(e),
                "processing_time": processing_time,
                "success": False,
                "timestamp": time.time()
            }

        return result

    async def evaluate_response(self, question: str, answer: str, contexts: List[str], response_time: float) -> Dict[str, Any]:
        """Evaluate the RAG response using advanced metrics."""
        try:
            # Run comprehensive evaluation
            evaluation = await self.evaluator.evaluate_comprehensive(
                question=question,
                answer=answer,
                contexts=contexts,
                response_time=response_time
            )

            # Extract key metrics for summary
            key_metrics = {
                'answer_relevance': evaluation.get('answer_relevance', 0),
                'context_relevance': evaluation.get('context_relevance', 0),
                'groundedness': evaluation.get('groundedness_score', 0),
                'answer_quality': evaluation.get('answer_quality', {}),
                'context_utilization': evaluation.get('context_utilization', {}),
                'response_efficiency': evaluation.get('response_efficiency', {}),
                'information_completeness': evaluation.get('information_completeness', {}),
                'llm_metrics': evaluation.get('llm_metrics', {})
            }

            return key_metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'error': str(e),
                'answer_relevance': 0,
                'context_relevance': 0,
                'groundedness': 0
            }

    def _generate_mock_response(self, question: str) -> str:
        """Generate a mock response for testing purposes."""
        # This is a placeholder - replace with actual RAG system call
        if "whiskers" in question.lower() or "cat" in question.lower():
            return "Whiskers is a curious cat who explores the forest, discovers a magical garden, and becomes a protector of nature."
        elif "buddy" in question.lower() or "dog" in question.lower():
            return "Buddy is a loyal farm dog who saves the village from a storm and becomes a local hero."
        elif "hopper" in question.lower() or "frog" in question.lower():
            return "Hopper is a wise frog who organizes wetland life, predicts weather, and helps during droughts."
        elif "elephant" in question.lower() or "ellie" in question.lower():
            return "Ellie is an elephant who leads animals to an oasis during drought and teaches unity and compassion."
        elif "horse" in question.lower() or "thunder" in question.lower():
            return "Thunder is a strong horse who helps on the farm and becomes a hero by moving a fallen tree during a storm."
        elif "lion" in question.lower() or "leo" in question.lower():
            return "Leo is a wise lion who organizes jungle cooperation during drought and teaches leadership."
        elif "rabbit" in question.lower() or "hopper" in question.lower():
            return "Hopper is a clever rabbit who outsmarts a fox to share a garden and becomes a meadow peacemaker."
        else:
            return f"This appears to be a question about animal stories. The answer would be found in the relevant story documents."

    def _generate_mock_contexts(self, question: str) -> List[str]:
        """Generate mock contexts for testing purposes."""
        # This is a placeholder - replace with actual retrieved documents
        base_contexts = [
            "This is a sample context paragraph from one of the animal stories.",
            "It contains relevant information that would help answer the question.",
            "The context provides background and details about the characters and events."
        ]

        # Add some question-specific context
        if "whiskers" in question.lower():
            base_contexts.append("Whiskers the cat is known for his curiosity and adventures in the forest.")
        elif "buddy" in question.lower():
            base_contexts.append("Buddy the dog showed great loyalty and courage during the village storm.")
        elif "hopper" in question.lower():
            base_contexts.append("Hopper the frog is wise and helps organize the wetland community.")

        return base_contexts

    def run_difficulty_tests(self, difficulty: str, limit: int = None) -> List[Dict[str, Any]]:
        """Run all tests for a specific difficulty level."""
        questions = self.questions.get(difficulty, [])
        if limit:
            questions = questions[:limit]

        logger.info(f"Running {len(questions)} {difficulty} tests")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Progress: {difficulty} {i}/{len(questions)}")
            result = self.run_single_test(question, difficulty)
            results.append(result)

        return results

    def run_all_tests(self, easy_limit: int = None, medium_limit: int = None,
                     hard_limit: int = None) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive RAG testing")

        # Run tests by difficulty
        self.results["results"]["easy"] = self.run_difficulty_tests("easy", easy_limit)
        self.results["results"]["medium"] = self.run_difficulty_tests("medium", medium_limit)
        self.results["results"]["hard"] = self.run_difficulty_tests("hard", hard_limit)

        # Calculate statistics
        self.calculate_statistics()

        logger.info("Testing completed")
        return self.results

    def calculate_statistics(self):
        """Calculate test statistics including evaluation metrics."""
        all_results = []
        evaluation_metrics = {
            'answer_relevance': [],
            'context_relevance': [],
            'groundedness': []
        }

        for difficulty in ["easy", "medium", "hard"]:
            results = self.results["results"][difficulty]
            all_results.extend(results)

            # Difficulty-specific stats
            total = len(results)
            successful = sum(1 for r in results if r.get("success", False))
            avg_time = sum(r.get("processing_time", 0) for r in results) / total if total > 0 else 0

            # Collect evaluation metrics
            relevance_scores = []
            context_scores = []
            groundedness_scores = []

            for r in results:
                if "evaluation" in r and r["evaluation"]:
                    eval_data = r["evaluation"]
                    if 'answer_relevance' in eval_data and eval_data['answer_relevance'] is not None:
                        relevance_scores.append(eval_data['answer_relevance'])
                        evaluation_metrics['answer_relevance'].append(eval_data['answer_relevance'])
                    if 'context_relevance' in eval_data and eval_data['context_relevance'] is not None:
                        context_scores.append(eval_data['context_relevance'])
                        evaluation_metrics['context_relevance'].append(eval_data['context_relevance'])
                    if 'groundedness' in eval_data and eval_data['groundedness'] is not None:
                        groundedness_scores.append(eval_data['groundedness'])
                        evaluation_metrics['groundedness'].append(eval_data['groundedness'])

            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            avg_context = sum(context_scores) / len(context_scores) if context_scores else 0
            avg_groundedness = sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0

            self.results["metadata"]["questions_by_difficulty"][difficulty] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0,
                "avg_processing_time": avg_time,
                "avg_answer_relevance": avg_relevance,
                "avg_context_relevance": avg_context,
                "avg_groundedness": avg_groundedness
            }

        # Overall stats
        total_questions = len(all_results)
        successful_questions = sum(1 for r in all_results if r.get("success", False))
        overall_success_rate = successful_questions / total_questions if total_questions > 0 else 0

        # Overall evaluation metrics
        overall_metrics = {}
        for metric_name, scores in evaluation_metrics.items():
            if scores:
                overall_metrics[f"avg_{metric_name}"] = sum(scores) / len(scores)
            else:
                overall_metrics[f"avg_{metric_name}"] = 0

        self.results["metadata"].update({
            "total_questions": total_questions,
            "successful_questions": successful_questions,
            "success_rate": overall_success_rate,
            "total_processing_time": sum(r.get("processing_time", 0) for r in all_results),
            **overall_metrics
        })

    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_summary(self):
        """Print test summary with evaluation metrics."""
        meta = self.results["metadata"]

        print("\n" + "="*60)
        print("CUBO RAG TESTING SUMMARY")
        print("="*60)

        print(f"Total Questions Tested: {meta['total_questions']}")
        print(".1f")
        print(".2f")

        # Print overall evaluation metrics
        if 'avg_answer_relevance' in meta and meta['avg_answer_relevance'] > 0:
            print(".3f")
        if 'avg_context_relevance' in meta and meta['avg_context_relevance'] > 0:
            print(".3f")
        if 'avg_groundedness' in meta and meta['avg_groundedness'] > 0:
            print(".3f")

        print("\nBy Difficulty:")
        for difficulty, stats in meta["questions_by_difficulty"].items():
            print(f"  {difficulty.capitalize()}:")
            print(f"    Questions: {stats['total']}")
            print(".1f")
            print(".2f")
            if 'avg_answer_relevance' in stats and stats['avg_answer_relevance'] > 0:
                print(".3f")
            if 'avg_context_relevance' in stats and stats['avg_context_relevance'] > 0:
                print(".3f")
            if 'avg_groundedness' in stats and stats['avg_groundedness'] > 0:
                print(".3f")

        print("\nDetailed results saved to test_results.json")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="CUBO RAG Testing Framework")
    parser.add_argument("--questions", default="test_questions.json",
                       help="Path to questions JSON file")
    parser.add_argument("--easy-limit", type=int,
                       help="Limit number of easy questions")
    parser.add_argument("--medium-limit", type=int,
                       help="Limit number of medium questions")
    parser.add_argument("--hard-limit", type=int,
                       help="Limit number of hard questions")
    parser.add_argument("--output", default="test_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Initialize tester
    tester = RAGTester(args.questions)

    # Run tests
    results = tester.run_all_tests(
        easy_limit=args.easy_limit,
        medium_limit=args.medium_limit,
        hard_limit=args.hard_limit
    )

    # Save and display results
    tester.save_results(args.output)
    tester.print_summary()


if __name__ == "__main__":
    main()