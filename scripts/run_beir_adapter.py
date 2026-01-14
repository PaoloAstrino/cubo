"""
BEIR Benchmark Adapter Script

This script runs BEIR (Benchmarking Information Retrieval) evaluations using the Cubo retrieval system.
It provides both full production retrieval and optimized batch retrieval modes for benchmarking.

Features:
- Full production mode: BM25 + dense retrieval + RRF fusion + reranking
- Optimized mode: Dense retrieval only (much faster for large-scale evaluation)
- Automatic index building from BEIR corpus
- BEIR evaluation metrics (NDCG, Recall, Precision)
- Comprehensive logging and metadata collection
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from cubo.adapters.beir_adapter import CuboBeirAdapter
from cubo.config import config
from cubo.config.settings import settings
from cubo.utils.logger import Logger

# Setup logging
logger = Logger()
log = logging.getLogger("beir_adapter")


def collect_benchmark_metadata(args, use_optimized: bool) -> Dict:
    """Collect all configuration details for reproducibility."""
    metadata = {
        "_benchmark_metadata": {
            "timestamp": datetime.now().isoformat(),
            "embedding_model": config.get("model_path", "unknown"),
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "index_dir": args.index_dir,
            "queries_file": args.queries,
            "retrieval_mode": "optimized_batch" if use_optimized else "full_production",
            "laptop_mode": config.is_laptop_mode(),
            "features": {
                "dense_search": True,
                "bm25_hybrid": not use_optimized,  # Only in full production mode
                "rrf_fusion": not use_optimized,  # Only in full production mode
                "reranking": not use_optimized,  # Skipped in optimized mode
                "sentence_window": not use_optimized,
            },
            "hyperparameters": {
                "bm25_k1": settings.retrieval.bm25_k1,
                "bm25_b": settings.retrieval.bm25_b,
                "rrf_k": settings.retrieval.rrf_k,
                "semantic_weight": settings.retrieval.semantic_weight_default,
                "bm25_weight": settings.retrieval.bm25_weight_default,
                "chunk_size": settings.chunking.chunk_size,
                "chunk_overlap_sentences": settings.chunking.chunk_overlap_sentences,
            },
        }
    }
    return metadata


def load_queries(queries_path: str, corpus_path: str = None) -> Dict[str, str]:
    """Load queries from BEIR queries.jsonl or queries.json"""
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        # Try loading as JSON first (dict format)
        try:
            data = json.load(f)
            if isinstance(data, dict):
                queries = data
        except json.JSONDecodeError:
            # Not JSON, try JSONL
            f.seek(0)
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    qid = item.get("_id", str(i))
                    text = item.get("text", item.get("query", ""))
                    if text:
                        queries[qid] = text
                except json.JSONDecodeError:
                    continue

    # If a corpus is provided, check if we need to resolve IDs
    if corpus_path and queries:
        first_query = next(iter(queries.values()), "")
        # Heuristic: if it looks like a hex/ID and is long enough
        import re
        if re.match(r"^[a-f0-9\-]+$", first_query) and len(first_query) >= 20:
            print(f"Detected ID-based queries (e.g., '{first_query}'). Resolving against corpus...")
            corpus_map = {}
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        cid = item.get("_id")
                        if cid:
                            corpus_map[str(cid)] = (item.get("title", "") + " " + item.get("text", "")).strip()
                    except json.JSONDecodeError:
                        continue
            
            resolved_count = 0
            for qid, query_val in queries.items():
                if query_val in corpus_map:
                    queries[qid] = corpus_map[query_val]
                    resolved_count += 1
            print(f"Resolved {resolved_count}/{len(queries)} queries.")
            
    return queries


def main():
    """Main function to run BEIR benchmark evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run BEIR benchmark using Cubo adapter")
    parser.add_argument("--corpus", type=str, help="Path to BEIR corpus.jsonl")
    parser.add_argument("--queries", type=str, required=True, help="Path to BEIR queries.jsonl")
    parser.add_argument(
        "--index-dir",
        type=str,
        default="results/beir_adapter_index",
        help="Directory for FAISS index",
    )
    parser.add_argument(
        "--output", type=str, default="results/beir_run.json", help="Output run file (JSON)"
    )
    parser.add_argument("--reindex", action="store_true", help="Rebuild index from corpus")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results per query")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for retrieval")
    parser.add_argument(
        "--limit", type=int, help="Limit number of documents to index (for testing)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run BEIR evaluation metrics (requires beir package)",
    )
    parser.add_argument(
        "--qrels", type=str, help="Path to qrels file (required if --evaluate is set)"
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized batch retrieval (much faster, recommended)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Run BM25-only ablation (force semantic_weight=0)",
    )
    parser.add_argument(
        "--laptop-mode",
        action="store_true",
        help="Enable laptop mode (lazy loading, no reranking, etc)",
    )
    parser.add_argument("--query-limit", type=int, help="Limit number of queries to process")

    args = parser.parse_args()

    # Apply configuration overrides based on command line flags
    if args.laptop_mode:
        log.info("Enabling laptop mode configuration...")
        config.apply_laptop_mode(force=True)

    # Apply BM25-only override for ablation studies
    if args.bm25_only:
        log.info("BM25-only mode: forcing semantic_weight=0 and dense_weight=0")
        # These keys exist in config or settings; apply runtime override
        config.set("model.semantic_weight", 0.0)
        config.set("model.dense_weight", 0.0)

    # Validate required arguments
    if args.reindex and not args.corpus:
        parser.error("--corpus is required when --reindex is set")

    if args.evaluate and not args.qrels:
        parser.error("--qrels is required when --evaluate is set")

    # Ensure results/logs exists and add a file handler for verbose logs
    from pathlib import Path
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"beir_adapter_{Path(args.output).stem}.log"
    try:
        fh = logging.FileHandler(str(logfile))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        # Avoid adding duplicate file handlers
        if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
            log.addHandler(fh)
        # Ensure console output is visible: add StreamHandler if missing
        if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(sh)
        log.info(f"Logging to {logfile} and console")
    except Exception as e:
        log.warning(f"Could not create log file handler: {e}")

    log.info("Initializing CuboBeirAdapter...")
    adapter = CuboBeirAdapter(index_dir=args.index_dir, lightweight=False)

    try:
        if args.reindex:
            log.info(f"Reindexing corpus from {args.corpus}...")
            adapter.index_corpus(corpus_path=args.corpus, index_dir=args.index_dir, limit=args.limit)
        else:
            log.info(f"Loading index from {args.index_dir}...")
            adapter.load_index(args.index_dir)
    except Exception as e:
        log.exception(f"Index operation failed for {args.index_dir}: {e}")
        print(f"Index operation failed: {e}. See log file: {logfile}", flush=True)
        sys.exit(1)
    log.info(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries, args.corpus)
    if args.query_limit:
        log.info(f"Limiting to first {args.query_limit} queries...")
        queries = dict(list(queries.items())[: args.query_limit])
    log.info(f"Loaded {len(queries)} queries")

    log.info(f"Running retrieval (top_k={args.top_k}, optimized={args.use_optimized})...")

    # Collect benchmark metadata for reproducibility
    metadata = collect_benchmark_metadata(args, args.use_optimized)

    # Use optimized retrieval if requested
    if args.use_optimized:
        results = adapter.retrieve_bulk_optimized(
            queries=queries, top_k=args.top_k, skip_reranker=True, batch_size=args.batch_size
        )
    else:
        results = adapter.retrieve_bulk(queries=queries, top_k=args.top_k)

    # Merge metadata with results (metadata first for visibility)
    output_data = {**metadata, **results}

    log.info(f"Saving run to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    if args.evaluate:
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval

            log.info("Running BEIR evaluation...")

            # Load qrels (ground truth relevance judgments)
            # BEIR expects qrels as Dict[str, Dict[str, int]]
            qrels = {}
            # Check if qrels is TSV or JSONL? BEIR usually provides TSV
            # Let's assume standard BEIR qrels format (TSV: query-id, corpus-id, score)
            # Or use beir utility if available.
            # For now, simple TSV loader
            with open(args.qrels, "r") as f:
                # skip header
                next(f)
                for line in f:
                    qid, did, score = line.strip().split("\t")
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = int(score)

            # Run evaluation using BEIR's evaluation framework
            evaluator = EvaluateRetrieval()
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])

            print("\n--- BEIR Evaluation Results ---")
            print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
            print(f"Recall@100: {recall['Recall@100']:.4f}")
            print(f"Precision@10: {precision['P@10']:.4f}")

            # Save metrics to separate file
            metrics_file = args.output.replace(".json", "_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(
                    {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision},
                    f,
                    indent=2,
                )
            log.info(f"Metrics saved to {metrics_file}")

        except ImportError:
            log.error("beir package not installed. Cannot run evaluation.")
        except Exception as e:
            log.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Unhandled exception in run_beir_adapter: %s", e)
        print(f"Unhandled error: {e}. See logs in results/logs/")
        sys.exit(2)
