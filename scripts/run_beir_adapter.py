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


def _load_queries_from_json(f):
    """Load queries from JSON format."""
    try:
        data = json.load(f)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return None


def _load_queries_from_jsonl(f):
    """Load queries from JSONL format."""
    queries = {}
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
    return queries


def _needs_id_resolution(queries):
    """Check if queries contain IDs that need resolution."""
    if not queries:
        return False
    first_query = next(iter(queries.values()), "")
    import re
    return re.match(r"^[a-f0-9\-]+$", first_query) and len(first_query) >= 20


def _build_corpus_map(corpus_path):
    """Build mapping of corpus IDs to text."""
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
    return corpus_map


def _resolve_query_ids(queries, corpus_map):
    """Resolve query IDs to actual text from corpus."""
    resolved_count = 0
    for qid, query_val in queries.items():
        if query_val in corpus_map:
            queries[qid] = corpus_map[query_val]
            resolved_count += 1
    print(f"Resolved {resolved_count}/{len(queries)} queries.")


def load_queries(queries_path: str, corpus_path: str = None) -> Dict[str, str]:
    """Load queries from BEIR queries.jsonl or queries.json"""
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = _load_queries_from_json(f)
        if queries is None:
            queries = _load_queries_from_jsonl(f)

    if corpus_path and _needs_id_resolution(queries):
        print(f"Detected ID-based queries. Resolving against corpus...")
        corpus_map = _build_corpus_map(corpus_path)
        _resolve_query_ids(queries, corpus_map)
            
    return queries


def _parse_beir_arguments():
    """Parse command line arguments for BEIR benchmark."""
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
    return parser.parse_args()


def _apply_config_overrides(args):
    """Apply configuration overrides based on command line flags."""
    if args.laptop_mode:
        log.info("Enabling laptop mode configuration...")
        config.apply_laptop_mode(force=True)

    if args.bm25_only:
        log.info("BM25-only mode: forcing semantic_weight=0 and dense_weight=0")
        config.set("model.semantic_weight", 0.0)
        config.set("model.dense_weight", 0.0)


def _validate_arguments(parser, args):
    """Validate required arguments."""
    if args.reindex and not args.corpus:
        parser.error("--corpus is required when --reindex is set")
    if args.evaluate and not args.qrels:
        parser.error("--qrels is required when --evaluate is set")


def _create_file_handler(logfile):
    """Create and configure file handler for logging."""
    fh = logging.FileHandler(str(logfile))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    return fh


def _create_console_handler():
    """Create and configure console handler for logging."""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    return sh


def _add_handlers_if_missing(logfile):
    """Add file and console handlers if not already present."""
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        log.addHandler(_create_file_handler(logfile))
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        log.addHandler(_create_console_handler())


def _setup_logging(args):
    """Setup file and console logging."""
    from pathlib import Path
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"beir_adapter_{Path(args.output).stem}.log"
    
    try:
        _add_handlers_if_missing(logfile)
        log.info(f"Logging to {logfile} and console")
    except Exception as e:
        log.warning(f"Could not create log file handler: {e}")
    
    return logfile


def _initialize_adapter_and_index(args, logfile):
    """Initialize adapter and load/build index."""
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
    
    return adapter


def _load_and_limit_queries(args):
    """Load queries and apply limit if specified."""
    log.info(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries, args.corpus)
    if args.query_limit:
        log.info(f"Limiting to first {args.query_limit} queries...")
        queries = dict(list(queries.items())[: args.query_limit])
    log.info(f"Loaded {len(queries)} queries")
    return queries


def _perform_retrieval(adapter, queries, args):
    """Perform retrieval using adapter with optimized or standard method."""
    log.info(f"Running retrieval (top_k={args.top_k}, optimized={args.use_optimized})...")
    
    if args.use_optimized:
        return adapter.retrieve_bulk_optimized(
            queries=queries, top_k=args.top_k, skip_reranker=True, batch_size=args.batch_size
        )
    else:
        return adapter.retrieve_bulk(queries=queries, top_k=args.top_k)


def _evaluate_and_save_metrics(args, results):
    """Evaluate results using BEIR framework and save metrics."""
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval

        log.info("Running BEIR evaluation...")

        # Load qrels (ground truth relevance judgments)
        qrels = {}
        with open(args.qrels, "r") as f:
            next(f)  # skip header
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
    except Exception as e:
        log.error(f"BEIR evaluation failed: {e}")


def main():
    """Main function to run BEIR benchmark evaluation."""
    args = _parse_beir_arguments()
    parser = argparse.ArgumentParser(description="Run BEIR benchmark using Cubo adapter")
    _apply_config_overrides(args)
    _validate_arguments(parser, args)
    logfile = _setup_logging(args)
    adapter = _initialize_adapter_and_index(args, logfile)
    queries = _load_and_limit_queries(args)

    # Collect benchmark metadata for reproducibility
    metadata = collect_benchmark_metadata(args, args.use_optimized)
    
    # Perform retrieval
    results = _perform_retrieval(adapter, queries, args)

    # Merge metadata with results (metadata first for visibility)
    output_data = {**metadata, **results}

    log.info(f"Saving run to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    if args.evaluate:
        _evaluate_and_save_metrics(args, results)

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
