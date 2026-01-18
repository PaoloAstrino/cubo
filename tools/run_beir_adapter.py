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
import time
from typing import Dict

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from evaluation.beir_adapter import CuboBeirAdapter
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
        print("Detected ID-based queries. Resolving against corpus...")
        corpus_map = _build_corpus_map(corpus_path)
        _resolve_query_ids(queries, corpus_map)
            
    return queries


def _parse_arguments():
    """Parse and validate command line arguments."""
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
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of random seeds for statistical validation")
    parser.add_argument("--hot-fraction", type=float, help="Override hot fraction (e.g. 0.2 to test IVF+PQ variance)")

    args = parser.parse_args()

    # Validate required arguments
    if args.reindex and not args.corpus:
        parser.error("--corpus is required when --reindex is set")

    if args.evaluate and not args.qrels:
        parser.error("--qrels is required when --evaluate is set")

    return args


def _apply_config_overrides(args):
    """Apply configuration overrides based on command line flags."""
    if args.laptop_mode:
        log.info("Enabling laptop mode configuration...")
        config.apply_laptop_mode(force=True)

    if args.bm25_only:
        log.info("BM25-only mode: forcing semantic_weight=0 and dense_weight=0")
        config.set("model.semantic_weight", 0.0)
        config.set("model.dense_weight", 0.0)


def _setup_logging(args):
    """Setup file and console logging handlers."""
    from pathlib import Path
    
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile = logs_dir / f"beir_adapter_{Path(args.output).stem}.log"
    
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(str(logfile))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            root_logger.addHandler(fh)
        
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            root_logger.addHandler(sh)
        
        log.info(f"Logging to {logfile} and console")
        return logfile
    except Exception as e:
        log.warning(f"Could not create log file handler: {e}")
        return None


def _initialize_adapter(args, logfile, seed=42):
    """Initialize and prepare the BEIR adapter with index."""
    log.info(f"Initializing CuboBeirAdapter (seed={seed})...")
    adapter = CuboBeirAdapter(
        index_dir=args.index_dir, 
        lightweight=False, 
        seed=seed,
        hot_fraction=args.hot_fraction if args.hot_fraction is not None else 0.2
    )

    try:
        if args.reindex:
            log.info(f"Reindexing corpus from {args.corpus}...")
            adapter.index_corpus(corpus_path=args.corpus, index_dir=args.index_dir, limit=args.limit, batch_size=args.batch_size)
        else:
            log.info(f"Loading index from {args.index_dir}...")
            adapter.load_index(args.index_dir)
    except Exception as e:
        log.exception(f"Index operation failed for {args.index_dir}: {e}")
        print(f"Index operation failed: {e}. See log file: {logfile}", flush=True)
        sys.exit(1)
    
    return adapter


def _load_and_limit_queries(args):
    """Load queries and optionally limit the number."""
    log.info(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries, args.corpus)
    
    # If evaluating, filter queries to only those in qrels
    if args.evaluate and args.qrels:
        log.info(f"Filtering queries to match qrels in {args.qrels}...")
        qrels_ids = set()
        with open(args.qrels, "r") as f:
            next(f) # Skip header
            for line in f:
                qid = line.strip().split("\t")[0]
                qrels_ids.add(qid)
        
        original_count = len(queries)
        queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels_ids}
        log.info(f"Filtered queries from {original_count} to {len(queries)} matching qrels.")

    if args.query_limit:
        log.info(f"Limiting to first {args.query_limit} queries...")
        queries = dict(list(queries.items())[: args.query_limit])
    
    log.info(f"Loaded {len(queries)} queries")
    return queries


def _run_retrieval(adapter, queries, args):
    """Execute retrieval using either optimized or full production mode."""
    log.info(f"Running retrieval (top_k={args.top_k}, optimized={args.use_optimized})...")
    
    if args.use_optimized:
        return adapter.retrieve_bulk_optimized(
            queries=queries, top_k=args.top_k, skip_reranker=True, batch_size=args.batch_size
        )
    else:
        return adapter.retrieve_bulk(queries=queries, top_k=args.top_k)


def _save_results(results, metadata, args):
    """Save retrieval results with metadata to output file."""
    output_data = {**metadata, **results}
    
    log.info(f"Saving run to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def _load_qrels(qrels_path: str) -> Dict:
    """Load BEIR qrels file in TSV format."""
    qrels = {}
    with open(qrels_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            qid, did, score = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(score)
    return qrels


def main():
    """Main function to run BEIR benchmark evaluation."""
    args = _parse_arguments()
    _apply_config_overrides(args)
    logfile = _setup_logging(args)
    
    queries = _load_and_limit_queries(args)
    metadata = collect_benchmark_metadata(args, args.use_optimized)
    
    all_metrics = []
    num_runs = args.num_seeds
    
    for i in range(num_runs):
        seed = 42 + i
        log.info(f"--- Starting Run {i+1}/{num_runs} with seed {seed} ---")
        
        # Initialize adapter with current seed
        adapter = _initialize_adapter(args, logfile, seed=seed)
        
        # Retrieval
        results = _run_retrieval(adapter, queries, args)

        # Evaluation (if requested)
        # defensive: ensure `metrics` is defined even when evaluation is skipped or fails
        metrics = None
        if args.evaluate:
            metrics = _calculate_metrics(results, args)
            if metrics:
                all_metrics.append(metrics)
        
        # Save individual run results
        if metrics is not None:
            run_metrics_file = args.output.replace(".json", f"_run_{i+1}_metrics.json")
            with open(run_metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            log.info(f"Saved metrics for run {i+1} to {run_metrics_file}")

        if num_runs == 1:
            _save_results(results, metadata, args)
            # Only attempt to print/write evaluation summary when we have metrics
            if args.evaluate and all_metrics:
                try:
                    _print_evaluation_results(all_metrics[0])
                    metrics_file = args.output.replace(".json", "_metrics.json")
                    with open(metrics_file, "w") as f:
                        json.dump(all_metrics[0], f, indent=2)
                except Exception as e:
                    log.warning(f"Could not print/save evaluation summary: {e}")

    if num_runs > 1 and all_metrics:
        _aggregate_and_report_metrics(all_metrics, args)

def _calculate_metrics(results, args):
    """Run BEIR evaluation and return metrics dict."""
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
        qrels = _load_qrels(args.qrels)
        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
        return {"ndcg": ndcg, "map": _map, "recall": recall, "precision": precision}
    except Exception as e:
        log.error(f"Metric calculation failed: {e}")
        return None

def _print_evaluation_results(metrics):
    print("\n--- BEIR Evaluation Results ---")
    print(f"NDCG@10: {metrics['ndcg']['NDCG@10']:.4f}")
    print(f"Recall@100: {metrics['recall']['Recall@100']:.4f}")
    print(f"Precision@10: {metrics['precision']['P@10']:.4f}")

def _aggregate_and_report_metrics(all_metrics, args):
    """Aggregate metrics from multiple runs and report mean ± std."""
    import numpy as np
    
    metric_keys = ["ndcg", "recall", "precision"]
    sub_keys = ["NDCG@10", "Recall@100", "P@10"]
    
    print(f"\n--- Statistical Results over {len(all_metrics)} runs ---")
    
    stats_out = {}
    
    for m_key in metric_keys:
        for s_key in sub_keys:
            if m_key in all_metrics[0] and s_key in all_metrics[0][m_key]:
                vals = [m[m_key][s_key] for m in all_metrics]
                mean = np.mean(vals)
                std = np.std(vals)
                print(f"{s_key}: {mean:.4f} ± {std:.4f}")
                stats_out[s_key] = {"mean": float(mean), "std": float(std), "raw": [float(v) for v in vals]}
    
    stats_file = args.output.replace(".json", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats_out, f, indent=2)
    log.info(f"Statistical results saved to {stats_file}")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Unhandled exception in run_beir_adapter: %s", e)
        print(f"Unhandled error: {e}. See logs in results/logs/")
        sys.exit(2)
