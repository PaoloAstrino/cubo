#!/usr/bin/env python
"""
BEIR Evaluation Script

Run evaluation on already-indexed documents using BEIR queries.
Usage:
    python scripts/run_beir_eval.py --test        # Quick 10-query test
    python scripts/run_beir_eval.py --full        # Full 648-query evaluation
    python scripts/run_beir_eval.py --max 100     # Custom limit
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run BEIR evaluation on existing index")
    parser.add_argument("--test", action="store_true", help="Quick 10-query test")
    parser.add_argument("--full", action="store_true", help="Full evaluation (648 queries)")
    parser.add_argument("--max", type=int, default=None, help="Max queries to run")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip RAGAS metrics (faster)")
    parser.add_argument("--index-dir", type=str, default="results/tonight_full/storage",
                        help="Path to vector store")
    parser.add_argument("--queries", type=str, default="data/beir_queries.jsonl",
                        help="Path to queries JSONL")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the FAISS index from BEIR corpus before running evaluation")
    parser.add_argument("--beir-corpus", type=str, default="data/beir/corpus.jsonl", help="Path to BEIR corpus.jsonl when reindexing")
    parser.add_argument("--force", action="store_true", help="Force run even if sanity checks fail")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (auto-generated if not provided)")
    args = parser.parse_args()

    # Determine max queries
    if args.test:
        max_queries = 10
        print("[MODE] Quick test: 10 queries")
    elif args.full:
        max_queries = None
        print("[MODE] Full evaluation: all queries")
    else:
        max_queries = args.max
        print(f"[MODE] Custom: {max_queries or 'all'} queries")

    # Auto-generate output filename
    if args.output:
        output_file = args.output
    else:
        suffix = f"_test{max_queries}" if max_queries else "_full"
        output_file = f"results/tonight_full/benchmark_beir{suffix}.json"

    print(f"\n[CONFIG]")
    print(f"  Index:   {args.index_dir}")
    print(f"  Queries: {args.queries}")
    print(f"  Output:  {output_file}")
    print(f"  RAGAS:   {'Disabled' if args.skip_ragas else 'Enabled'}")
    print()

    # Configure CUBO
    from cubo.config import config
    config.set("vector_store_path", args.index_dir)
    config.set("document_cache_size", 0)
    config.set("llm.model_name", "llama3.2:latest")

    # Verify Ollama is running only if RAGAS is enabled
    if not args.skip_ragas:
        try:
            import ollama
            models = [m.model for m in ollama.list().models]
            print(f"[OK] Ollama running with models: {models}")
        except Exception as e:
            print(f"[ERROR] Ollama not reachable: {e}")
            print("  Start Ollama first: ollama serve")
            sys.exit(1)

    # Optionally rebuild FAISS index from BEIR corpus
    if args.reindex:
        print("[ACTION] Rebuilding FAISS index from BEIR corpus using CUBO ingestion...")
        import subprocess
        builder = PROJECT_ROOT / 'scripts' / 'reindex_beir_with_cubo.py'
        cmd = [sys.executable, str(builder), '--corpus', args.beir_corpus, '--index-dir', args.index_dir]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            print("[OK] Rebuilt index using CUBO ingestion")
        except Exception as e:
            print(f"[ERROR] Failed to reindex using CUBO pipeline: {e}")
            print("Falling back to building FAISS index via scripts/build_beir_faiss_index.py")
            builder2 = PROJECT_ROOT / 'scripts' / 'build_beir_faiss_index.py'
            cmd2 = [sys.executable, str(builder2), '--corpus', args.beir_corpus, '--index-dir', args.index_dir]
            try:
                subprocess.check_call(cmd2)
                print("[OK] Rebuilt index with FAISS builder")
            except Exception as e2:
                print(f"[ERROR] Failed to build index with FAISS builder too: {e2}")
                sys.exit(1)

    # Run evaluation
    from benchmarks.reproduce.batch_eval import run_batch_eval
    import time

    print("\n[STARTING] Evaluation...")
    start = time.time()
    summary = run_batch_eval(
        queries_file=args.queries,
        output_file=output_file,
        max_queries=max_queries,
        skip_ragas=args.skip_ragas,
        force_run=args.force,
    )
    elapsed = time.time() - start

    print(f"\n[DONE] {summary.total_queries} queries in {elapsed:.1f}s")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
