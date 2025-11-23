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

from src.cubo.main import CUBOApp
from src.cubo.evaluation.metrics import AdvancedEvaluator, IRMetricsEvaluator, GroundTruthLoader
from src.cubo.evaluation.perf_utils import sample_latency, sample_memory, log_hardware_metadata
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

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available, LLM-based evaluations will be disabled")

class RAGTester:
    """Comprehensive RAG testing framework."""

    def __init__(self, questions_file: str = "test_questions.json", data_folder: str = "data",
                 ground_truth_file: str = None, mode: str = "full"):
        """Initialize the tester with question data and CUBO system."""
        self.questions_file = questions_file
        self.questions = self.load_questions()
        self.data_folder = data_folder
        self.mode = mode  # 'full', 'retrieval-only', 'ingestion-only'
        
        # Load ground truth for IR metrics if provided
        self.ground_truth = None
        if ground_truth_file:
            try:
                if ground_truth_file.endswith('.json'):
                    self.ground_truth = GroundTruthLoader.load_custom_format(ground_truth_file)
                else:
                    self.ground_truth = GroundTruthLoader.load_beir_format(ground_truth_file)
                logger.info(f"Loaded ground truth for {len(self.ground_truth)} questions")
            except Exception as e:
                logger.error(f"Failed to load ground truth: {e}")

        # Initialize evaluator
        if OLLAMA_AVAILABLE:
            self.evaluator = AdvancedEvaluator(ollama_client=ollama.Client())
            logger.info("AdvancedEvaluator initialized with Ollama client for LLM-based metrics")
        else:
            self.evaluator = AdvancedEvaluator()
            logger.info("AdvancedEvaluator initialized without LLM client (LLM-based metrics disabled)")

        # Initialize IR metrics evaluator
        self.ir_evaluator = IRMetricsEvaluator()

        # Initialize CUBO system (skip for ingestion-only mode)
        self.cubo_app = None
        if mode != "ingestion-only":
            self._initialize_cubo_system()

        # Capture hardware metadata
        self.hardware_metadata = log_hardware_metadata()
        logger.info(f"Hardware: {self.hardware_metadata['cpu']['model']}, "
                   f"{self.hardware_metadata['ram']['total_gb']:.1f}GB RAM")

        self.results = {
            "metadata": {
                "test_run_timestamp": time.time(),
                "total_questions": 0,
                "questions_by_difficulty": {},
                "success_rate": 0.0,
                "mode": mode,
                "hardware": self.hardware_metadata
            },
            "results": {
                "easy": [],
                "medium": [],
                "hard": []
            }
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
                if isinstance(chunk, dict) and 'text' in chunk:
                    document_texts.append(chunk['text'])
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
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {data['metadata']['total_questions']} questions")
            return data['questions']
        except Exception as e:
            logger.error(f"Failed to load questions: {e}")
            return {"easy": [], "medium": [], "hard": []}

    def run_single_test(self, question: str, difficulty: str, question_id: str = None,
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Run a single question test with real RAG evaluation."""
        logger.info(f"Testing [{difficulty}]: {question[:50]}...")

        start_time = time.time()

        try:
            if not self.cubo_app:
                raise Exception("CUBO system not initialized")

            # Get actual retrieved contexts from CUBO with latency measurement
            retrieval_metrics = sample_latency(
                self.cubo_app.retriever.retrieve_top_documents,
                question,
                samples=1
            )
            contexts = self.cubo_app.retriever.retrieve_top_documents(question)
            
            # Extract document IDs for IR metrics
            retrieved_ids = []
            for ctx in contexts:
                if isinstance(ctx, dict):
                    doc_id = ctx.get('id') or ctx.get('doc_id') or ctx.get('chunk_id')
                    if doc_id:
                        retrieved_ids.append(str(doc_id))

            # Compute IR metrics if ground truth available
            ir_metrics = {}
            if self.ground_truth and question_id:
                ir_metrics = self.ir_evaluator.evaluate_retrieval(
                    question_id,
                    retrieved_ids,
                    self.ground_truth,
                    k_values=k_values
                )

            # For retrieval-only mode, skip generation
            if self.mode == "retrieval-only":
                processing_time = time.time() - start_time
                
                result = {
                    "question": question,
                    "question_id": question_id,
                    "difficulty": difficulty,
                    "retrieved_ids": retrieved_ids,
                    "contexts": contexts,
                    "retrieval_latency": retrieval_metrics,
                    "ir_metrics": ir_metrics,
                    "processing_time": processing_time,
                    "success": True,
                    "timestamp": time.time()
                }
                return result

            # Full RAG mode: generate response
            context_texts = [ctx.get('document', '') if isinstance(ctx, dict) else str(ctx) for ctx in contexts]
            context_text = "\n".join(context_texts)
            
            # Measure generation latency
            generation_start = time.time()
            response = self.cubo_app.generator.generate_response(question, context_text)
            generation_time = time.time() - generation_start

            processing_time = time.time() - start_time

            # Sample memory during evaluation
            memory_metrics = sample_memory()

            # Evaluate the response using AdvancedEvaluator from metrics.py
            evaluation_results = asyncio.run(self.evaluate_response(question, response, context_texts, processing_time))

            result = {
                "question": question,
                "question_id": question_id,
                "difficulty": difficulty,
                "response": response,
                "retrieved_ids": retrieved_ids,
                "contexts": contexts,
                "retrieval_latency": retrieval_metrics,
                "generation_time": generation_time,
                "processing_time": processing_time,
                "memory": memory_metrics,
                "ir_metrics": ir_metrics,
                "evaluation": evaluation_results,
                "success": True,
                "timestamp": time.time()
            }

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Test failed for question: {question[:50]}... Error: {e}")
            result = {
                "question": question,
                "question_id": question_id,
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

    def run_difficulty_tests(self, difficulty: str, limit: int = None,
                            k_values: List[int] = [5, 10, 20]) -> List[Dict[str, Any]]:
        """Run all tests for a specific difficulty level."""
        questions = self.questions.get(difficulty, [])
        if limit:
            questions = questions[:limit]

        logger.info(f"Running {len(questions)} {difficulty} tests")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Progress: {difficulty} {i}/{len(questions)}")
            # Generate question_id if not provided
            question_id = f"{difficulty}_{i}"
            result = self.run_single_test(question, difficulty, question_id=question_id, k_values=k_values)
            results.append(result)

        return results

    def run_all_tests(self, easy_limit: int = None, medium_limit: int = None,
                     hard_limit: int = None, k_values: List[int] = [5, 10, 20]) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        logger.info(f"Starting comprehensive RAG testing (mode: {self.mode})")

        # Run tests by difficulty
        self.results["results"]["easy"] = self.run_difficulty_tests("easy", easy_limit, k_values=k_values)
        self.results["results"]["medium"] = self.run_difficulty_tests("medium", medium_limit, k_values=k_values)
        self.results["results"]["hard"] = self.run_difficulty_tests("hard", hard_limit, k_values=k_values)

        # Calculate statistics
        self.calculate_statistics()

        logger.info("Testing completed")
        return self.results

    def calculate_statistics(self):
        """Calculate test statistics including evaluation and IR metrics."""
        all_results = []
        evaluation_metrics = {
            'answer_relevance': [],
            'context_relevance': [],
            'groundedness': []
        }
        ir_metrics_aggregated = {}

        for difficulty in ["easy", "medium", "hard"]:
            results = self.results["results"][difficulty]
            all_results.extend(results)

            # Difficulty-specific stats
            total = len(results)
            successful = sum(1 for r in results if r.get("success", False))
            avg_time = sum(r.get("processing_time", 0) for r in results) / total if total > 0 else 0
            
            # Collect latency stats
            retrieval_latencies = []
            for r in results:
                if "retrieval_latency" in r and r["retrieval_latency"]:
                    retrieval_latencies.append(r["retrieval_latency"].get("p50_ms", 0))
            
            avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0

            # Collect IR metrics
            ir_stats = {}
            for r in results:
                if "ir_metrics" in r and r["ir_metrics"]:
                    for metric_name, values in r["ir_metrics"].items():
                        if isinstance(values, dict):
                            for k, score in values.items():
                                key = f"{metric_name}_{k}"
                                if key not in ir_stats:
                                    ir_stats[key] = []
                                ir_stats[key].append(score)

            # Average IR metrics
            avg_ir_metrics = {}
            for key, scores in ir_stats.items():
                avg_ir_metrics[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0

            # Collect evaluation metrics (for full RAG mode)
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
                "avg_retrieval_latency_p50_ms": avg_retrieval_latency,
                "avg_answer_relevance": avg_relevance,
                "avg_context_relevance": avg_context,
                "avg_groundedness": avg_groundedness,
                **avg_ir_metrics
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
        
        # Overall IR metrics
        overall_ir_metrics = {}
        for r in all_results:
            if "ir_metrics" in r and r["ir_metrics"]:
                for metric_name, values in r["ir_metrics"].items():
                    if isinstance(values, dict):
                        for k, score in values.items():
                            key = f"{metric_name}_{k}"
                            if key not in overall_ir_metrics:
                                overall_ir_metrics[key] = []
                            overall_ir_metrics[key].append(score)
        
        for key, scores in overall_ir_metrics.items():
            overall_metrics[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0

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
        """Print test summary with evaluation and IR metrics."""
        meta = self.results["metadata"]

        print("\n" + "="*60)
        print(f"CUBO RAG TESTING SUMMARY (Mode: {meta.get('mode', 'full')})")
        print("="*60)

        print(f"Total Questions Tested: {meta['total_questions']}")
        print(f"Success Rate: {meta['success_rate']*100:.1f}%")
        print(f"Total Processing Time: {meta['total_processing_time']:.2f}s")

        # Print overall IR metrics
        print("\nOverall IR Metrics:")
        for key, value in meta.items():
            if key.startswith('avg_recall_at_k'):
                k = key.split('_')[-1]
                print(f"  Recall@{k}: {value:.3f}")
        for key, value in meta.items():
            if key.startswith('avg_ndcg_at_k'):
                k = key.split('_')[-1]
                print(f"  nDCG@{k}: {value:.3f}")

        # Print overall evaluation metrics (full RAG mode)
        if meta.get('mode') == 'full':
            print("\nOverall RAG Metrics:")
            if 'avg_answer_relevance' in meta and meta['avg_answer_relevance'] > 0:
                print(f"  Answer Relevance: {meta['avg_answer_relevance']:.3f}")
            if 'avg_context_relevance' in meta and meta['avg_context_relevance'] > 0:
                print(f"  Context Relevance: {meta['avg_context_relevance']:.3f}")
            if 'avg_groundedness' in meta and meta['avg_groundedness'] > 0:
                print(f"  Groundedness: {meta['avg_groundedness']:.3f}")

        print("\nBy Difficulty:")
        for difficulty, stats in meta["questions_by_difficulty"].items():
            print(f"  {difficulty.capitalize()}:")
            print(f"    Questions: {stats['total']}")
            print(f"    Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"    Avg Processing Time: {stats['avg_processing_time']:.2f}s")
            
            if 'avg_retrieval_latency_p50_ms' in stats:
                print(f"    Avg Retrieval Latency (p50): {stats['avg_retrieval_latency_p50_ms']:.1f}ms")
            
            # Show IR metrics per difficulty
            for key, value in stats.items():
                if key.startswith('avg_recall_at_k'):
                    k = key.split('_')[-1]
                    print(f"    Recall@{k}: {value:.3f}")
            
            # Show RAG metrics per difficulty (full mode)
            if meta.get('mode') == 'full':
                if 'avg_answer_relevance' in stats and stats['avg_answer_relevance'] > 0:
                    print(f"    Answer Relevance: {stats['avg_answer_relevance']:.3f}")
                if 'avg_groundedness' in stats and stats['avg_groundedness'] > 0:
                    print(f"    Groundedness: {stats['avg_groundedness']:.3f}")

        # Hardware summary
        if 'hardware' in meta:
            hw = meta['hardware']
            print("\nHardware Configuration:")
            print(f"  CPU: {hw['cpu']['model']}")
            print(f"  RAM: {hw['ram']['total_gb']:.1f} GB")
            if hw['gpu'].get('available'):
                print(f"  GPU: {hw['gpu']['device_name']} ({hw['gpu']['vram_total_gb']:.1f} GB VRAM)")

        print("\nDetailed results saved to test_results.json")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="CUBO RAG Testing Framework")
    parser.add_argument("--questions", default="test_questions.json",
                       help="Path to questions JSON file")
    parser.add_argument("--data-folder", default="data",
                       help="Path to data folder containing documents")
    parser.add_argument("--ground-truth", default=None,
                       help="Path to ground truth file (BeIR format or custom JSON)")
    parser.add_argument("--mode", default="full", choices=["full", "retrieval-only", "ingestion-only"],
                       help="Testing mode: full RAG, retrieval-only, or ingestion-only")
    parser.add_argument("--k-values", default="5,10,20",
                       help="Comma-separated K values for IR metrics (default: 5,10,20)")
    parser.add_argument("--easy-limit", type=int,
                       help="Limit number of easy questions")
    parser.add_argument("--medium-limit", type=int,
                       help="Limit number of medium questions")
    parser.add_argument("--hard-limit", type=int,
                       help="Limit number of hard questions")
    parser.add_argument("--output", default="test_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    # Initialize tester with data folder
    tester = RAGTester(
        args.questions,
        args.data_folder,
        ground_truth_file=args.ground_truth,
        mode=args.mode
    )

    # Run tests
    results = tester.run_all_tests(
        easy_limit=args.easy_limit,
        medium_limit=args.medium_limit,
        hard_limit=args.hard_limit,
        k_values=k_values
    )

    # Save and display results
    tester.save_results(args.output)
    tester.print_summary()


if __name__ == "__main__":
    main()