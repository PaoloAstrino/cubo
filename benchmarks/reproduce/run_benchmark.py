"""
One-Click Benchmark Runner

Orchestrates the full benchmark pipeline:
1. Prepare corpus (optional)
2. Ingest documents
3. Build index
4. Generate queries (optional)
5. Run evaluation
6. Output results

Usage:
    python run_benchmark.py --smoke          # Quick test
    python run_benchmark.py --full           # Full benchmark
    python run_benchmark.py --skip-prep      # Skip dataset prep
"""
import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import psutil

# Add benchmarks to path for sibling imports
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))


@dataclass
class PipelineStats:
    """Pipeline execution statistics."""
    corpus_prep_sec: float = 0
    ingestion_sec: float = 0
    indexing_sec: float = 0
    query_gen_sec: float = 0
    evaluation_sec: float = 0
    total_sec: float = 0
    peak_ram_gb: float = 0
    corpus_chunks: int = 0
    queries_run: int = 0


def get_ram_gb() -> float:
    """Get current process RAM in GB."""
    return psutil.Process().memory_info().rss / (1024 ** 3)


def get_system_ram_gb() -> float:
    """Get total system RAM usage in GB."""
    return psutil.virtual_memory().used / (1024 ** 3)


def run_with_timing(name: str, func, *args, **kwargs):
    """Run function with timing and RAM monitoring."""
    print(f"\n{'='*50}")
    print(f"[STEP] {name}")
    print(f"{'='*50}")
    
    ram_before = get_system_ram_gb()
    start = time.time()
    
    result = func(*args, **kwargs)
    
    elapsed = time.time() - start
    ram_after = get_system_ram_gb()
    ram_peak = max(ram_before, ram_after)
    
    print(f"[DONE] {name}: {elapsed:.1f}s, RAM: {ram_peak:.2f} GB")
    
    return result, elapsed, ram_peak


def run_full_pipeline(
    corpus_dir: Path = Path("data/benchmark_corpus"),
    queries_file: Path = Path("data/benchmark_queries.jsonl"),
    results_dir: Path = Path("results"),
    domains: list = ["legal"],
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
    skip_prep: bool = False,
    skip_ingest: bool = False,
    skip_query_gen: bool = False,
    skip_ragas: bool = False,
    config_path: Optional[Path] = None,
    seed: int = 42,
) -> PipelineStats:
    """
    Run the full benchmark pipeline.
    
    Args:
        corpus_dir: Directory for corpus data
        queries_file: Output file for generated queries
        results_dir: Directory for results
        domains: Domains to benchmark
        max_docs: Limit docs per domain
        max_queries: Limit queries to run
        skip_prep: Skip corpus preparation
        skip_ingest: Skip ingestion/indexing
        skip_query_gen: Skip query generation
        skip_ragas: Skip RAGAS evaluation
        config_path: CUBO config path
        seed: Random seed
        
    Returns:
        Pipeline execution statistics
    """
    stats = PipelineStats()
    peak_rams = []
    total_start = time.time()
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare corpus
    if not skip_prep and not skip_ingest:
        from prepare_ultradomain import prepare_ultradomain
        
        result, elapsed, ram = run_with_timing(
            "Corpus Preparation",
            prepare_ultradomain,
            domains=domains,
            output_dir=corpus_dir,
            max_docs_per_domain=max_docs,
        )
        stats.corpus_prep_sec = elapsed
        stats.corpus_chunks = sum(v.get("chunks", 0) for v in result.values())
        peak_rams.append(ram)
    else:
        print("\n[SKIP] Corpus preparation (--skip-prep or --skip-ingest)")
    
    # Step 2: Initialize CUBO and build index
    from cubo.core import CuboCore
    from cubo.config import config
    
    # Isolate benchmark storage (don't use default ./faiss_store)
    bench_data_dir = results_dir / "storage"
    bench_data_dir.mkdir(parents=True, exist_ok=True)
    config.set("vector_store_path", str(bench_data_dir))
    config.set("document_cache_size", 0) # disable cache for clean test
    
    print(f"\n[STEP] Initializing CUBO in isolated environment: {bench_data_dir}")
    cubo = CuboCore()
    cubo.initialize_components()
    
    # Step 3: Ingest documents
    if not skip_ingest:
        # The corpus files are JSONL, we need to convert them to txt for the doc_loader
        corpus_files = list(Path(corpus_dir).glob("*_corpus.jsonl"))
        
        if corpus_files:
            # Create a temp folder with the corpus as txt files for ingestion
            import tempfile
            import shutil
            
            temp_corpus_dir = Path(tempfile.mkdtemp(prefix="cubo_bench_"))
            print(f"\n  Preparing corpus in {temp_corpus_dir}")
            
            doc_count = 0
            for corpus_file in corpus_files:
                with open(corpus_file, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if max_docs and doc_count >= max_docs:
                            break
                        try:
                            doc = json.loads(line)
                            text = doc.get("text", "")
                            if text:
                                doc_file = temp_corpus_dir / f"doc_{doc_count:05d}.txt"
                                doc_file.write_text(text, encoding="utf-8")
                                doc_count += 1
                        except json.JSONDecodeError:
                            continue
            
            print(f"  Created {doc_count} document files")
            
            # Explicitly load using doc_loader to verify loading works
            print("  Loading documents with DocumentLoader...")
            if not cubo.doc_loader:
                 from cubo.ingestion.document_loader import DocumentLoader
                 cubo.doc_loader = DocumentLoader()
                 
            loaded_chunks = cubo.doc_loader.load_documents_from_folder(str(temp_corpus_dir))
            print(f"  Loader returned {len(loaded_chunks)} chunks")
            
            if loaded_chunks:
                # Add to retriever
                result, elapsed, ram = run_with_timing(
                    "Adding Docs to Index",
                    cubo.retriever.add_documents,
                    loaded_chunks
                )
                stats.ingestion_sec = elapsed
                peak_rams.append(ram)
            else:
                 print("  [ERROR] DocumentLoader returned 0 chunks!")
            
            # Cleanup temp dir
            shutil.rmtree(temp_corpus_dir, ignore_errors=True)
        else:
            print("\n[SKIP] No corpus files found, skipping ingestion")
    else:
        print("\n[SKIP] Ingestion and indexing (--skip-ingest)")
        # Verify index exists
        if not (bench_data_dir / "documents.db").exists():
            print(f"\n[WARN] Index directory {bench_data_dir} seems empty. Ensure you ran without --skip-ingest first.")
    
    
    # Step 4: Generate queries
    if not skip_query_gen:
        from generate_queries import generate_queries
        
        result, elapsed, ram = run_with_timing(
            "Query Generation",
            generate_queries,
            corpus_dir=corpus_dir,
            output_file=queries_file,
            domains=domains,
            n_chunks=min(50, max_docs or 50),
            seed=seed,
        )
        stats.query_gen_sec = elapsed
        peak_rams.append(ram)
    elif not queries_file.exists() and (max_queries and max_queries < 50):
        # Fallback for smoke tests: Create dummy queries if file doesn't exist
        print("\n[WARN] Queries file missing and generation skipped. Creating dummy queries for smoke test.")
        queries_file.parent.mkdir(parents=True, exist_ok=True)
        with open(queries_file, "w", encoding="utf-8") as f:
            dummies = [
                {"query": "What is the capital of France?", "domain": "general"},
                {"query": "Explain quantum entanglement", "domain": "physics"},
                {"query": "Legal definition of contract", "domain": "legal"},
                {"query": "Symptoms of flu", "domain": "medical"},
                {"query": "How to bake a cake", "domain": "cooking"},
            ]
            for q in dummies:
                f.write(json.dumps(q) + "\n")
    else:
        print("\n[SKIP] Query generation (--skip-query-gen)")
    
    # Step 5: Run evaluation
    from batch_eval import run_batch_eval
    
    if not queries_file.exists():
        print(f"\n[ERROR] Queries file not found: {queries_file}")
        print("  Run without --skip-query-gen or ensure file exists.")
        return stats
    
    output_file = results_dir / "benchmark_results.json"
    summary, elapsed, ram = run_with_timing(
        "Batch Evaluation",
        run_batch_eval,
        queries_file=queries_file,
        output_file=output_file,
        max_queries=max_queries,
        skip_ragas=skip_ragas,
    )
    stats.evaluation_sec = elapsed
    stats.queries_run = summary.total_queries
    peak_rams.append(ram)
    
    # Finalize stats
    stats.total_sec = time.time() - total_start
    stats.peak_ram_gb = max(peak_rams) if peak_rams else 0
    
    # Save pipeline stats
    stats_file = results_dir / "pipeline_stats.json"
    with open(stats_file, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    
    print_final_report(stats, results_dir)
    
    return stats


def _ingest_corpus(cubo, corpus_files: list) -> int:
    """Ingest corpus files into CUBO."""
    all_docs = []
    for corpus_file in corpus_files:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                # Format for CuboCore.add_documents()
                all_docs.append({
                    "text": doc["text"],
                    "file_path": doc.get("id", "benchmark_doc"),
                    "metadata": {"domain": doc.get("domain", "unknown")},
                })
    
    # Add all documents at once
    cubo.add_documents(all_docs)
    print(f"  Ingested {len(all_docs)} documents")
    return len(all_docs)


def print_final_report(stats: PipelineStats, results_dir: Path):
    """Print final benchmark report."""
    print("\n")
    print("=" * 60)
    print(" CUBO BENCHMARK REPORT")
    print("=" * 60)
    print(f"""
Pipeline Timing:
  Corpus Prep:    {stats.corpus_prep_sec:>8.1f} s
  Ingestion:      {stats.ingestion_sec:>8.1f} s
  Indexing:       {stats.indexing_sec:>8.1f} s
  Query Gen:      {stats.query_gen_sec:>8.1f} s
  Evaluation:     {stats.evaluation_sec:>8.1f} s
  ─────────────────────────────
  TOTAL:          {stats.total_sec:>8.1f} s ({stats.total_sec/60:.1f} min)

Resource Usage:
  Peak RAM:       {stats.peak_ram_gb:>8.2f} GB
  Corpus Chunks:  {stats.corpus_chunks:>8d}
  Queries Run:    {stats.queries_run:>8d}

Results saved to: {results_dir}
""")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="One-click CUBO benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (5 docs, 10 queries)
  python run_benchmark.py --smoke
  
  # Full benchmark on legal domain
  python run_benchmark.py --full --domains legal
  
  # Skip data prep (use existing corpus)
  python run_benchmark.py --skip-prep --skip-query-gen
        """
    )
    
    # Presets
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--full", action="store_true", help="Full benchmark")
    
    # Data options
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/benchmark_corpus"))
    parser.add_argument("--queries-file", type=Path, default=Path("data/benchmark_queries.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--domains", nargs="+", default=["legal"])
    
    # Limits
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    
    # Skip options
    parser.add_argument("--skip-prep", action="store_true", help="Skip corpus prep")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion/indexing (use existing index)")
    parser.add_argument("--skip-query-gen", action="store_true", help="Skip query generation")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS eval")
    
    # Config
    parser.add_argument("--config", type=Path, default=None, help="CUBO config path")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Apply presets
    if args.smoke:
        args.max_docs = 5
        args.max_queries = 10
        args.skip_ragas = True
        print("[MODE] Smoke test (5 docs, 10 queries, no RAGAS)")
    elif args.full:
        args.max_docs = None
        args.max_queries = None
        print("[MODE] Full benchmark")
    
    try:
        run_full_pipeline(
            corpus_dir=args.corpus_dir,
            queries_file=args.queries_file,
            results_dir=args.results_dir,
            domains=args.domains,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            skip_prep=args.skip_prep,
            skip_ingest=args.skip_ingest,
            skip_query_gen=args.skip_query_gen,
            skip_ragas=args.skip_ragas,
            config_path=args.config,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\n[ABORT] Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
