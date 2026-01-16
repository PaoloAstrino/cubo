"""
Generation Quality Evaluation Script using Local RAGAS.

This script evaluates end-to-end RAG quality (Faithfulness, Answer Relevancy, 
Context Precision) using CUBO's local LLM as the judge model.

Usage:
    python tools/run_generation_eval.py --dataset politics --num-samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time

from cubo.core import CuboCore
from cubo.utils.logger import setup_logger
from evaluation.ragas_evaluator import run_ragas_evaluation

logger = setup_logger(__name__)


def load_test_queries(dataset_name: str, num_samples: int = 50) -> List[Dict[str, Any]]:
    """Load test queries from BEIR or UltraDomain datasets."""
    # Map dataset names to file paths
    dataset_paths = {
        "politics": "data/ultradomain/politics_queries.jsonl",
        "legal": "data/ultradomain/legal_queries.jsonl",
        "nfcorpus": "data/beir/nfcorpus/queries.jsonl",
        "fiqa": "data/beir/fiqa/queries.jsonl",
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_paths.keys())}")
    
    queries_path = Path(dataset_paths[dataset_name])
    if not queries_path.exists():
        logger.warning(f"Dataset file not found: {queries_path}. Using synthetic fallback.")
        return generate_synthetic_queries(dataset_name, num_samples)
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            queries.append({
                "question": data.get("query", data.get("question", "")),
                "ground_truth": data.get("answer", data.get("ground_truth", "")),
                "query_id": data.get("id", f"{dataset_name}_{i}")
            })
    
    return queries


def generate_synthetic_queries(dataset_name: str, num_samples: int) -> List[Dict[str, Any]]:
    """Generate synthetic queries for testing when real data is unavailable."""
    logger.info(f"Generating {num_samples} synthetic queries for {dataset_name}")
    
    templates = {
        "politics": [
            ("What is the role of the European Parliament?", "The European Parliament is the legislative body of the EU."),
            ("How are EU laws created?", "EU laws are created through the ordinary legislative procedure."),
        ],
        "legal": [
            ("What is GDPR Article 5?", "GDPR Article 5 defines principles for processing personal data."),
            ("What are data subject rights?", "Data subjects have rights to access, rectification, and erasure."),
        ],
        "nfcorpus": [
            ("What causes diabetes?", "Diabetes is caused by insufficient insulin production or insulin resistance."),
            ("How is hypertension treated?", "Hypertension is treated with lifestyle changes and medication."),
        ],
    }
    
    base_templates = templates.get(dataset_name, templates["politics"])
    queries = []
    
    for i in range(num_samples):
        template_idx = i % len(base_templates)
        question, answer = base_templates[template_idx]
        queries.append({
            "question": f"{question} (variant {i})",
            "ground_truth": answer,
            "query_id": f"synthetic_{dataset_name}_{i}"
        })
    
    return queries


def run_rag_pipeline(cubo: CuboCore, question: str) -> Dict[str, Any]:
    """Run full RAG pipeline: retrieve + generate."""
    # Retrieve contexts
    retrieval_results = cubo.retrieve(query=question, top_k=5)
    contexts = [chunk["text"] for chunk in retrieval_results["chunks"]]
    
    # Generate answer
    context_str = "\n\n".join(contexts)
    answer = cubo.generate(query=question, context=context_str)
    
    return {
        "contexts": contexts,
        "answer": answer,
        "retrieval_time": retrieval_results.get("latency_ms", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run generation quality evaluation with local RAGAS")
    parser.add_argument("--dataset", type=str, default="politics", 
                       help="Dataset to evaluate (politics, legal, nfcorpus, fiqa)")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of queries to evaluate")
    parser.add_argument("--output", type=str, default="results/generation_eval",
                       help="Output directory for results")
    parser.add_argument("--laptop-mode", action="store_true",
                       help="Enable laptop mode (resource-constrained)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🚀 Starting generation evaluation on {args.dataset} ({args.num_samples} samples)")
    
    # Initialize CUBO
    logger.info("Initializing CUBO core...")
    cubo = CuboCore(laptop_mode=args.laptop_mode)
    
    # Load test queries
    logger.info(f"Loading test queries from {args.dataset}...")
    test_queries = load_test_queries(args.dataset, args.num_samples)
    logger.info(f"Loaded {len(test_queries)} queries")
    
    # Run RAG pipeline for each query
    logger.info("Running RAG pipeline...")
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []
    
    start_time = time.time()
    for i, query_data in enumerate(test_queries):
        logger.info(f"Processing query {i+1}/{len(test_queries)}: {query_data['question'][:50]}...")
        
        try:
            result = run_rag_pipeline(cubo, query_data["question"])
            
            questions.append(query_data["question"])
            contexts_list.append(result["contexts"])
            answers.append(result["answer"])
            ground_truths.append(query_data["ground_truth"])
            
        except Exception as e:
            logger.error(f"Failed to process query {i}: {e}")
            continue
    
    total_time = time.time() - start_time
    logger.info(f"RAG pipeline completed in {total_time:.1f}s ({total_time/len(test_queries):.2f}s per query)")
    
    # Run RAGAS evaluation
    logger.info("Running RAGAS evaluation with local judge...")
    try:
        ragas_start = time.time()
        scores = run_ragas_evaluation(
            questions=questions,
            contexts=contexts_list,
            answers=answers,
            ground_truths=ground_truths,
        )
        ragas_time = time.time() - ragas_start
        logger.info(f"RAGAS evaluation completed in {ragas_time:.1f}s")
        
        # Print results
        print("\n" + "="*60)
        print(f"📊 RAGAS Evaluation Results ({args.dataset})")
        print("="*60)
        for metric, value in scores.items():
            print(f"  {metric:25s}: {value:.4f}")
        print("="*60)
        
        # Save results
        results = {
            "dataset": args.dataset,
            "num_samples": len(questions),
            "ragas_scores": {k: float(v) for k, v in scores.items()},
            "avg_rag_latency_s": total_time / len(test_queries),
            "ragas_eval_time_s": ragas_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        output_file = output_dir / f"{args.dataset}_ragas_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
