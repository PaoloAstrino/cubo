"""
Batch Evaluation with RAGAS

Evaluates CUBO RAG responses using RAGAS metrics.
Calculates win-rate comparisons for paper benchmarks.
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

import psutil
from tqdm import tqdm

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class EvalResult:
    """Single evaluation result."""
    query: str
    answer: str
    contexts: List[str]
    domain: str
    latency_ms: float
    ram_mb: float
    # RAGAS scores (filled after evaluation)
    faithfulness: Optional[float] = None
    relevancy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None


@dataclass 
class BenchmarkSummary:
    """Aggregated benchmark results."""
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    peak_ram_mb: float
    avg_faithfulness: float
    avg_relevancy: float
    avg_precision: float
    avg_recall: float


def get_ram_mb() -> float:
    """Get current RAM usage in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_query(cubo_core, query: str) -> tuple[str, List[str], float, float]:
    """
    Run a single query through CUBO and measure performance.
    
    Returns: (answer, contexts, latency_ms, ram_mb)
    """
    ram_before = get_ram_mb()
    start = time.perf_counter()
    
    # Run retrieval
    results = cubo_core.query_retrieve(query, top_k=10)
    contexts = [r.get("text", r.get("content", "")) for r in results]
    context_str = "\n\n".join(contexts)
    
    # Generate response
    answer = cubo_core.generate_response_safe(query, context_str)
    
    latency_ms = (time.perf_counter() - start) * 1000
    ram_after = get_ram_mb()
    
    return answer, contexts, latency_ms, max(ram_before, ram_after)


def evaluate_with_ragas(results: List[EvalResult]) -> List[EvalResult]:
    """Apply RAGAS metrics to results."""
    if not RAGAS_AVAILABLE or not DATASETS_AVAILABLE:
        print("Warning: RAGAS not available, skipping metric calculation")
        return results
    
    # Prepare dataset for RAGAS
    data = {
        "question": [r.query for r in results],
        "answer": [r.answer for r in results],
        "contexts": [r.contexts for r in results],
        "ground_truth": ["" for _ in results],  # We don't have ground truth
    }
    dataset = Dataset.from_dict(data)
    
    try:
        # Run RAGAS evaluation
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        
        # Map scores back to results
        for i, r in enumerate(results):
            r.faithfulness = scores.get("faithfulness", [0])[i] if isinstance(scores.get("faithfulness"), list) else scores.get("faithfulness", 0)
            r.relevancy = scores.get("answer_relevancy", [0])[i] if isinstance(scores.get("answer_relevancy"), list) else scores.get("answer_relevancy", 0)
            r.precision = scores.get("context_precision", [0])[i] if isinstance(scores.get("context_precision"), list) else scores.get("context_precision", 0)
            
    except Exception as e:
        print(f"Warning: RAGAS evaluation failed: {e}")
    
    return results


def calculate_summary(results: List[EvalResult]) -> BenchmarkSummary:
    """Calculate aggregate statistics."""
    latencies = sorted([r.latency_ms for r in results])
    
    return BenchmarkSummary(
        total_queries=len(results),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p50_latency_ms=latencies[len(latencies) // 2] if latencies else 0,
        p99_latency_ms=latencies[int(len(latencies) * 0.99)] if latencies else 0,
        peak_ram_mb=max(r.ram_mb for r in results) if results else 0,
        avg_faithfulness=sum(r.faithfulness or 0 for r in results) / len(results) if results else 0,
        avg_relevancy=sum(r.relevancy or 0 for r in results) / len(results) if results else 0,
        avg_precision=sum(r.precision or 0 for r in results) / len(results) if results else 0,
        avg_recall=sum(r.recall or 0 for r in results) / len(results) if results else 0,
    )


def run_batch_eval(
    queries_file: Path,
    output_file: Path,
    config_path: Optional[Path] = None,
    max_queries: Optional[int] = None,
    skip_ragas: bool = False,
) -> BenchmarkSummary:
    """
    Run batch evaluation on queries.
    
    Args:
        queries_file: JSONL file with queries
        output_file: Output file for results
        config_path: Optional CUBO config path
        max_queries: Limit number of queries (for testing)
        skip_ragas: Skip RAGAS evaluation
        
    Returns:
        Aggregated benchmark summary
    """
    # Import CUBO (delayed to avoid import errors)
    from cubo.core import CuboCore
    
    # Load queries
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    
    if max_queries:
        queries = queries[:max_queries]
    
    print(f"Loaded {len(queries)} queries")
    
    # Initialize CUBO
    print("Initializing CUBO...")
    cubo = CuboCore()
    cubo.initialize_components()
    
    # Run queries
    results = []
    peak_ram = 0
    
    for q in tqdm(queries, desc="Running queries"):
        answer, contexts, latency, ram = run_query(cubo, q["query"])
        peak_ram = max(peak_ram, ram)
        
        results.append(EvalResult(
            query=q["query"],
            answer=answer,
            contexts=contexts,
            domain=q.get("domain", "unknown"),
            latency_ms=latency,
            ram_mb=ram,
        ))
    
    # Apply RAGAS
    if not skip_ragas:
        print("Running RAGAS evaluation...")
        results = evaluate_with_ragas(results)
    
    # Calculate summary
    summary = calculate_summary(results)
    
    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print_summary(summary)
    
    return summary


def print_summary(summary: BenchmarkSummary):
    """Print formatted summary table."""
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Queries:     {summary.total_queries}")
    print(f"Avg Latency:       {summary.avg_latency_ms:.1f} ms")
    print(f"P50 Latency:       {summary.p50_latency_ms:.1f} ms")
    print(f"P99 Latency:       {summary.p99_latency_ms:.1f} ms")
    print(f"Peak RAM:          {summary.peak_ram_mb:.1f} MB")
    print("-" * 50)
    print("RAGAS Scores (0-1):")
    print(f"  Faithfulness:    {summary.avg_faithfulness:.3f}")
    print(f"  Relevancy:       {summary.avg_relevancy:.3f}")
    print(f"  Precision:       {summary.avg_precision:.3f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation with RAGAS")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/benchmark_queries.jsonl"),
        help="Input queries JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark_results.json"),
        help="Output results JSON file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="CUBO config file path"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit queries (for testing)"
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS evaluation"
    )
    
    args = parser.parse_args()
    
    run_batch_eval(
        queries_file=args.queries,
        output_file=args.output,
        config_path=args.config,
        max_queries=args.max_queries,
        skip_ragas=args.skip_ragas,
    )


if __name__ == "__main__":
    main()
