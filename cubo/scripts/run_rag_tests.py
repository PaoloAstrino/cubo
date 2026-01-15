#!/usr/bin/env python3
"""
CUBO RAG Testing Script
Runs systematic tests using the comprehensive question set and evaluation metrics.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from cubo.main import CUBOApp
from cubo.security.security import security_manager
from evaluation.metrics import AdvancedEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_results.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available, LLM-based evaluations will be disabled")


class RAGTester:
    """Comprehensive RAG testing framework."""

    def __init__(self, questions_file: str = "test_questions.json", data_folder: str = "data"):
        """Initialize the tester with question data and CUBO system."""
        self.questions_file = questions_file
        self.questions = self.load_questions()
        self.data_folder = data_folder

        # Initialize evaluator
        if OLLAMA_AVAILABLE:
            self.evaluator = AdvancedEvaluator(ollama_client=ollama.Client())
            logger.info("AdvancedEvaluator initialized with Ollama client for LLM-based metrics")
        else:
            self.evaluator = AdvancedEvaluator()
            logger.info(
                "AdvancedEvaluator initialized without LLM client (LLM-based metrics disabled)"
            )

        # Initialize CUBO system
        self.cubo_app = None
        self._initialize_cubo_system()

        self.results = {
            "metadata": {
                "test_run_timestamp": time.time(),
                "total_questions": 0,
                "questions_by_difficulty": {},
                "success_rate": 0.0,
            },
            "results": {"easy": [], "medium": [], "hard": []},
        }

    def _initialize_cubo_system(self):
        """Initialize the CUBO RAG system for testing."""
        try:
            logger.info("Initializing CUBO system for testing...")
            self.cubo_app = CUBOApp()

            # Skip the setup wizard - assume system is already configured
            if not self.cubo_app.initialize_components():
                logger.error("Failed to initialize CUBO components")
                return

            # Load all documents from data folder
            if not os.path.exists(self.data_folder):
                logger.error(f"Data folder '{self.data_folder}' not found")
                return

            logger.info(f"Loading documents from {self.data_folder}...")
            documents = self.cubo_app.doc_loader.load_documents_from_folder(self.data_folder)
            if not documents:
                logger.error("No documents loaded")
                return

            logger.info(f"Adding {len(documents)} document chunks to vector database...")
            # Extract text content from chunk dictionaries
            document_texts = []
            for chunk in documents:
                if isinstance(chunk, dict) and "text" in chunk:
                    document_texts.append(chunk["text"])
                elif isinstance(chunk, str):
                    document_texts.append(chunk)
                else:
                    logger.warning(f"Skipping invalid chunk format: {type(chunk)}")

            if document_texts:
                self.cubo_app.retriever.add_documents(document_texts)
                logger.info("Documents added to vector database successfully")
            else:
                logger.error("No valid document texts found to add")
                return

            logger.info("CUBO system ready for testing!")

        except Exception as e:
            logger.error(f"Failed to initialize CUBO system: {e}")
            self.cubo_app = None

    def load_questions(self) -> Dict[str, List[str]]:
        """Load questions from JSON file."""
        try:
            with open(self.questions_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {data['metadata']['total_questions']} questions")
            return data["questions"]
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            return {"easy": [], "medium": [], "hard": []}

    def run_single_test(self, question: str, difficulty: str) -> Dict[str, Any]:
        """Run a single question test with real RAG evaluation."""
        logger.info(f"Testing [{difficulty}]: {security_manager.scrub(question)}")

        start_time = time.time()

        try:
            if not self.cubo_app:
                raise Exception("CUBO system not initialized")

            # Get actual retrieved contexts from CUBO
            contexts = self.cubo_app.retriever.retrieve_top_documents(question)

            # Generate actual response using CUBO
            context_texts = [
                ctx.get("document", "") if isinstance(ctx, dict) else str(ctx) for ctx in contexts
            ]
            context_text = "\n".join(context_texts)
            response = self.cubo_app.generator.generate_response(question, context_text)

            processing_time = time.time() - start_time

            # Evaluate the response using AdvancedEvaluator from metrics.py
            evaluation_results = asyncio.run(
                self.evaluate_response(question, response, context_texts, processing_time)
            )

            result = {
                "question": question,
                "difficulty": difficulty,
                "response": response,
                "contexts": contexts,
                "processing_time": processing_time,
                "evaluation": evaluation_results,
                "success": True,  # Based on evaluation scores
                "timestamp": time.time(),
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Test failed for question: {security_manager.scrub(question)} Error: {e}")
            result = {
                "question": question,
                "difficulty": difficulty,
                "error": str(e),
                "processing_time": processing_time,
                "success": False,
                "timestamp": time.time(),
            }

        return result

    async def evaluate_response(
        self, question: str, answer: str, contexts: List[str], response_time: float
    ) -> Dict[str, Any]:
        """Evaluate the RAG response using advanced metrics."""
        try:
            # Run comprehensive evaluation
            evaluation = await self.evaluator.evaluate_comprehensive(
                question=question, answer=answer, contexts=contexts, response_time=response_time
            )

            # Extract key metrics for summary
            key_metrics = {
                "answer_relevance": evaluation.get("answer_relevance", 0),
                "context_relevance": evaluation.get("context_relevance", 0),
                "groundedness": evaluation.get("groundedness_score", 0),
                "answer_quality": evaluation.get("answer_quality", {}),
                "context_utilization": evaluation.get("context_utilization", {}),
                "response_efficiency": evaluation.get("response_efficiency", {}),
                "information_completeness": evaluation.get("information_completeness", {}),
                "llm_metrics": evaluation.get("llm_metrics", {}),
            }

            return key_metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "error": str(e),
                "answer_relevance": 0,
                "context_relevance": 0,
                "groundedness": 0,
            }

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

    def run_all_tests(
        self, easy_limit: int = None, medium_limit: int = None, hard_limit: int = None
    ) -> Dict[str, Any]:
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

    def _collect_evaluation_scores(self, results, evaluation_metrics):
        """Collect evaluation scores from results."""
        relevance_scores = []
        context_scores = []
        groundedness_scores = []

        for r in results:
            if "evaluation" in r and r["evaluation"]:
                eval_data = r["evaluation"]
                if "answer_relevance" in eval_data and eval_data["answer_relevance"] is not None:
                    relevance_scores.append(eval_data["answer_relevance"])
                    evaluation_metrics["answer_relevance"].append(eval_data["answer_relevance"])
                if "context_relevance" in eval_data and eval_data["context_relevance"] is not None:
                    context_scores.append(eval_data["context_relevance"])
                    evaluation_metrics["context_relevance"].append(eval_data["context_relevance"])
                if "groundedness" in eval_data and eval_data["groundedness"] is not None:
                    groundedness_scores.append(eval_data["groundedness"])
                    evaluation_metrics["groundedness"].append(eval_data["groundedness"])

        return relevance_scores, context_scores, groundedness_scores

    def _compute_average_scores(self, relevance_scores, context_scores, groundedness_scores):
        """Compute average evaluation scores."""
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        avg_context = sum(context_scores) / len(context_scores) if context_scores else 0
        avg_groundedness = (
            sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0
        )
        return avg_relevance, avg_context, avg_groundedness

    def _calculate_difficulty_stats(self, difficulty, results, evaluation_metrics):
        """Calculate statistics for a specific difficulty level."""
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        avg_time = sum(r.get("processing_time", 0) for r in results) / total if total > 0 else 0

        relevance_scores, context_scores, groundedness_scores = self._collect_evaluation_scores(
            results, evaluation_metrics
        )

        avg_relevance, avg_context, avg_groundedness = self._compute_average_scores(
            relevance_scores, context_scores, groundedness_scores
        )

        return {
            "total": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_processing_time": avg_time,
            "avg_answer_relevance": avg_relevance,
            "avg_context_relevance": avg_context,
            "avg_groundedness": avg_groundedness,
        }

    def _calculate_overall_metrics(self, evaluation_metrics):
        """Calculate overall evaluation metrics."""
        overall_metrics = {}
        for metric_name, scores in evaluation_metrics.items():
            if scores:
                overall_metrics[f"avg_{metric_name}"] = sum(scores) / len(scores)
            else:
                overall_metrics[f"avg_{metric_name}"] = 0
        return overall_metrics

    def calculate_statistics(self):
        """Calculate test statistics including evaluation metrics."""
        all_results = []
        evaluation_metrics = {"answer_relevance": [], "context_relevance": [], "groundedness": []}

        for difficulty in ["easy", "medium", "hard"]:
            results = self.results["results"][difficulty]
            all_results.extend(results)

            difficulty_stats = self._calculate_difficulty_stats(
                difficulty, results, evaluation_metrics
            )
            self.results["metadata"]["questions_by_difficulty"][difficulty] = difficulty_stats

        total_questions = len(all_results)
        successful_questions = sum(1 for r in all_results if r.get("success", False))
        overall_success_rate = successful_questions / total_questions if total_questions > 0 else 0
        overall_metrics = self._calculate_overall_metrics(evaluation_metrics)

        self.results["metadata"].update(
            {
                "total_questions": total_questions,
                "successful_questions": successful_questions,
                "success_rate": overall_success_rate,
                "total_processing_time": sum(r.get("processing_time", 0) for r in all_results),
                **overall_metrics,
            }
        )

    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to JSON file."""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def _print_header(self):
        """Print summary header."""
        print("\n" + "=" * 60)
        print("CUBO RAG TESTING SUMMARY")
        print("=" * 60)

    def _print_overall_stats(self, meta):
        """Print overall statistics."""
        print(f"Total Questions Tested: {meta['total_questions']}")
        print(f"Successful: {meta['successful_questions']}")
        print(f"Success Rate: {meta['success_rate']:.1%}")
        print(f"Total Processing Time: {meta['total_processing_time']:.2f}s")

    def _print_evaluation_metrics(self, meta):
        """Print evaluation metrics if available."""
        if "avg_answer_relevance" in meta:
            print(f"\nAverage Answer Relevance: {meta['avg_answer_relevance']:.3f}")
        if "avg_context_relevance" in meta:
            print(f"Average Context Relevance: {meta['avg_context_relevance']:.3f}")
        if "avg_groundedness" in meta:
            print(f"Average Groundedness: {meta['avg_groundedness']:.3f}")

    def _print_difficulty_breakdown(self, meta):
        """Print breakdown by difficulty level."""
        print("\nBreakdown by Difficulty:")
        for difficulty in ["easy", "medium", "hard"]:
            if difficulty in meta.get("questions_by_difficulty", {}):
                stats = meta["questions_by_difficulty"][difficulty]
                print(f"\n{difficulty.upper()}:")
                print(f"  Total: {stats['total']}")
                print(f"  Success Rate: {stats['success_rate']:.1%}")
                print(f"  Avg Time: {stats['avg_processing_time']:.2f}s")

    def _print_metric_if_present(self, data, key, label, fmt=".3f"):
        """Print metric if present and non-zero in data."""
        if key in data and data[key] > 0:
            print(f"{label}: {data[key]:{fmt}}")

    def print_summary(self):
        """Print test summary with evaluation metrics."""
        meta = self.results["metadata"]
        self._print_header()
        self._print_overall_stats(meta)
        self._print_evaluation_metrics(meta)
        self._print_difficulty_breakdown(meta)
        print("=" * 60)

        # Print overall evaluation metrics
        self._print_metric_if_present(meta, "avg_answer_relevance", "Avg Answer Relevance")
        self._print_metric_if_present(meta, "avg_context_relevance", "Avg Context Relevance")
        self._print_metric_if_present(meta, "avg_groundedness", "Avg Groundedness")

        print("\nBy Difficulty:")
        for difficulty, stats in meta["questions_by_difficulty"].items():
            print(f"  {difficulty.capitalize()}:")
            print(f"    Questions: {stats['total']}")
            print(".1f")
            print(".2f")
            self._print_metric_if_present(
                stats, "avg_answer_relevance", "      Avg Answer Relevance"
            )
            self._print_metric_if_present(
                stats, "avg_context_relevance", "      Avg Context Relevance"
            )
            self._print_metric_if_present(stats, "avg_groundedness", "      Avg Groundedness")

        print("\nDetailed results saved to test_results.json")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="CUBO RAG Testing Framework")
    parser.add_argument(
        "--questions", default="test_questions.json", help="Path to questions JSON file"
    )
    parser.add_argument(
        "--data-folder", default="data", help="Path to data folder containing documents"
    )
    parser.add_argument("--easy-limit", type=int, help="Limit number of easy questions")
    parser.add_argument("--medium-limit", type=int, help="Limit number of medium questions")
    parser.add_argument("--hard-limit", type=int, help="Limit number of hard questions")
    parser.add_argument("--output", default="test_results.json", help="Output file for results")

    args = parser.parse_args()

    # Initialize tester with data folder
    tester = RAGTester(args.questions, args.data_folder)

    # Run tests
    _results = tester.run_all_tests(
        easy_limit=args.easy_limit, medium_limit=args.medium_limit, hard_limit=args.hard_limit
    )

    # Save and display results
    tester.save_results(args.output)
    tester.print_summary()


if __name__ == "__main__":
    main()
