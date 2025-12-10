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
    # Retriever returns "document" key, fallback to "text"/"content" for compatibility
    contexts = [r.get("document", r.get("text", r.get("content", ""))) for r in results]
    context_str = "\n\n".join(contexts)
    
    # Generate response
    answer = cubo_core.generate_response_safe(query, context_str)
    
    latency_ms = (time.perf_counter() - start) * 1000
    ram_after = get_ram_mb()
    
    return answer, contexts, latency_ms, max(ram_before, ram_after)


def evaluate_with_ragas(results: List[EvalResult]) -> List[EvalResult]:
    """Apply RAGAS metrics to results using local Ollama."""
    if not RAGAS_AVAILABLE or not DATASETS_AVAILABLE:
        print("Warning: RAGAS not available, skipping metric calculation")
        return results
    
    # Configure RAGAS to use Ollama
    try:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Use local Ollama model for RAGAS evaluation
        llm = ChatOllama(model="llama3.2:latest", temperature=0)
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Use local embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        
    except ImportError as e:
        print(f"Warning: Could not configure Ollama for RAGAS: {e}")
        print("Install with: pip install langchain-ollama langchain-huggingface")
        return results
    
    # Truncate contexts to avoid token limits (max ~2000 chars each)
    MAX_CONTEXT_LEN = 2000
    truncated_contexts = []
    for r in results:
        truncated = [c[:MAX_CONTEXT_LEN] if c else "" for c in r.contexts]
        truncated_contexts.append(truncated)
    
    # Prepare dataset for RAGAS (new column names in recent versions)
    data = {
        "user_input": [r.query for r in results],
        "response": [r.answer for r in results],
        "retrieved_contexts": truncated_contexts,
        "reference": ["N/A" for _ in results],  # We don't have ground truth
    }
    dataset = Dataset.from_dict(data)
    
    try:
        # Run RAGAS evaluation with local LLM (skip context_precision since it needs reference)
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        # Convert to pandas for easier access
        scores_df = scores.to_pandas()
        
        # Map scores back to results
        for i, r in enumerate(results):
            if i < len(scores_df):
                r.faithfulness = float(scores_df.iloc[i].get("faithfulness", 0) or 0)
                r.relevancy = float(scores_df.iloc[i].get("answer_relevancy", 0) or 0)
                r.precision = 0.0  # context_precision needs ground truth
            
    except Exception as e:
        print(f"Warning: RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def calculate_summary(results: List[EvalResult]) -> BenchmarkSummary:
    """Calculate aggregate statistics."""
    import math
    
    def safe_avg(values):
        """Calculate average, ignoring NaN values."""
        valid = [v for v in values if v is not None and not math.isnan(v)]
        return sum(valid) / len(valid) if valid else 0.0
    
    latencies = sorted([r.latency_ms for r in results])
    
    return BenchmarkSummary(
        total_queries=len(results),
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
        p50_latency_ms=latencies[len(latencies) // 2] if latencies else 0,
        p99_latency_ms=latencies[int(len(latencies) * 0.99)] if latencies else 0,
        peak_ram_mb=max(r.ram_mb for r in results) if results else 0,
        avg_faithfulness=safe_avg([r.faithfulness for r in results]),
        avg_relevancy=safe_avg([r.relevancy for r in results]),
        avg_precision=safe_avg([r.precision for r in results]),
        avg_recall=safe_avg([r.recall for r in results]),
    )


def run_batch_eval(
    queries_file: Path,
    output_file: Path,
    config_path: Optional[Path] = None,
    max_queries: Optional[int] = None,
    skip_ragas: bool = False,
    force_run: bool = False,
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

    # Sanity check the index to detect obvious mismatches (e.g., full docs used as contexts,
    # or a single doc appearing as top1 for most queries). This helps prevent meaningless runs.
    def sanity_check_index(cubo, queries, n_samples: int = 30, max_avg_len: int = 10000, max_dup_ratio: float = 0.5):
        """Run quick retrieval for `n_samples` queries and detect index anomalies.

        Returns a tuple (issues, diagnostics)
        """
        if not queries:
            return [], {}

        n = min(len(queries), n_samples)
        top1_texts = []
        lengths = []
        domains = []
        for i in range(n):
            q = queries[i]
            res = cubo.query_retrieve(q["query"], top_k=1)
            if not res:
                top1 = ""
            else:
                top1 = res[0].get("document", res[0].get("text", res[0].get("content", "")))
            top1_texts.append(top1)
            lengths.append(len(top1))
            domains.append(res[0].get("metadata", {}).get("domain") if res and res[0].get("metadata") else None)

        unique_top1 = len(set(top1_texts))
        dup_ratio = 1.0 - (unique_top1 / n)
        avg_len = sum(lengths) / n

        issues = []
        if avg_len > max_avg_len:
            issues.append(f"Avg context length too long ({avg_len:.0f} chars > {max_avg_len})")
        if dup_ratio > max_dup_ratio:
            issues.append(f"High top1 duplication ({dup_ratio*100:.0f}% identical top1 contexts)")

        diagnostics = {
            "sample_n": n,
            "avg_top1_len": avg_len,
            "top1_dup_ratio": dup_ratio,
            "unique_top1": unique_top1,
            "domains_sample": domains[:10],
        }

        return issues, diagnostics

    # Run sanity checks unless forcing the run
    issues, diagnostics = sanity_check_index(cubo, queries)
    if issues and not force_run:
        print("\n[ERROR] Index sanity check failed. Aborting benchmark run.")
        print("  Issues:")
        for it in issues:
            print(f"    - {it}")
        print("  Diagnostics:")
        for k, v in diagnostics.items():
            print(f"    - {k}: {v}")
        print('\n  If you are confident the index is correct, re-run with force_run=True')
        # Return an empty summary to indicate early abort
        return BenchmarkSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even if sanity checks fail",
    )
    
    args = parser.parse_args()
    
    run_batch_eval(
        queries_file=args.queries,
        output_file=args.output,
        config_path=args.config,
        max_queries=args.max_queries,
        skip_ragas=args.skip_ragas,
        force_run=args.force,
    )


if __name__ == "__main__":
    main()
