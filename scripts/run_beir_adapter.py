import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubo.adapters.beir_adapter import CuboBeirAdapter
from cubo.utils.logger import Logger
from cubo.config import config
from cubo.config.settings import settings
from datetime import datetime

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
                "rrf_fusion": not use_optimized,   # Only in full production mode
                "reranking": not use_optimized,    # Skipped in optimized mode
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

def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries from BEIR queries.jsonl or queries.json"""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        # Try loading as JSON first (dict format)
        try:
            data = json.load(f)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # Not JSON, try JSONL
            pass
        
        # Reset file pointer and try JSONL
        f.seek(0)
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                qid = item.get('_id', str(i))
                text = item.get('text', item.get('query', ''))
                if text:
                    queries[qid] = text
            except json.JSONDecodeError:
                continue
    return queries

def main():
    parser = argparse.ArgumentParser(description="Run BEIR benchmark using Cubo adapter")
    parser.add_argument("--corpus", type=str, help="Path to BEIR corpus.jsonl")
    parser.add_argument("--queries", type=str, required=True, help="Path to BEIR queries.jsonl")
    parser.add_argument("--index-dir", type=str, default="results/beir_adapter_index", help="Directory for FAISS index")
    parser.add_argument("--output", type=str, default="results/beir_run.json", help="Output run file (JSON)")
    parser.add_argument("--reindex", action="store_true", help="Rebuild index from corpus")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results per query")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for retrieval")
    parser.add_argument("--limit", type=int, help="Limit number of documents to index (for testing)")
    parser.add_argument("--evaluate", action="store_true", help="Run BEIR evaluation metrics (requires beir package)")
    parser.add_argument("--qrels", type=str, help="Path to qrels file (required if --evaluate is set)")
    parser.add_argument("--use-optimized", action="store_true", help="Use optimized batch retrieval (much faster, recommended)")
    parser.add_argument("--laptop-mode", action="store_true", help="Enable laptop mode (lazy loading, no reranking, etc)")
    parser.add_argument("--query-limit", type=int, help="Limit number of queries to process")
    
    args = parser.parse_args()
    
    if args.laptop_mode:
        log.info("Enabling laptop mode configuration...")
        config.apply_laptop_mode(force=True)
    
    # Validate args
    if args.reindex and not args.corpus:
        parser.error("--corpus is required when --reindex is set")
        
    if args.evaluate and not args.qrels:
        parser.error("--qrels is required when --evaluate is set")

    log.info("Initializing CuboBeirAdapter...")
    adapter = CuboBeirAdapter(index_dir=args.index_dir, lightweight=False)
    
    if args.reindex:
        log.info(f"Reindexing corpus from {args.corpus}...")
        adapter.index_corpus(
            corpus_path=args.corpus,
            index_dir=args.index_dir,
            limit=args.limit
        )
    else:
        log.info(f"Loading index from {args.index_dir}...")
        adapter.load_index(args.index_dir)
        
    log.info(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    if args.query_limit:
        log.info(f"Limiting to first {args.query_limit} queries...")
        queries = dict(list(queries.items())[:args.query_limit])
    log.info(f"Loaded {len(queries)} queries")
    
    log.info(f"Running retrieval (top_k={args.top_k}, optimized={args.use_optimized})...")
    
    # Collect benchmark metadata for reproducibility
    metadata = collect_benchmark_metadata(args, args.use_optimized)
    
    # Use optimized retrieval if requested
    if args.use_optimized:
        results = adapter.retrieve_bulk_optimized(
            queries=queries,
            top_k=args.top_k,
            skip_reranker=True,
            batch_size=args.batch_size
        )
    else:
        results = adapter.retrieve_bulk(
            queries=queries,
            top_k=args.top_k
        )
    
    # Merge metadata with results (metadata first for visibility)
    output_data = {**metadata, **results}
    
    log.info(f"Saving run to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    if args.evaluate:
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
            log.info("Running BEIR evaluation...")
            
            # Load qrels
            # BEIR expects qrels as Dict[str, Dict[str, int]]
            qrels = {}
            # Check if qrels is TSV or JSONL? BEIR usually provides TSV
            # Let's assume standard BEIR qrels format (TSV: query-id, corpus-id, score)
            # Or use beir utility if available.
            # For now, simple TSV loader
            with open(args.qrels, 'r') as f:
                # skip header
                next(f) 
                for line in f:
                    qid, did, score = line.strip().split('\t')
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = int(score)
            
            evaluator = EvaluateRetrieval()
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 10, 100])
            
            print("\n--- BEIR Evaluation Results ---")
            print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
            print(f"Recall@100: {recall['Recall@100']:.4f}")
            print(f"Precision@10: {precision['P@10']:.4f}")
            
            # Save metrics
            metrics_file = args.output.replace('.json', '_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump({'ndcg': ndcg, 'map': _map, 'recall': recall, 'precision': precision}, f, indent=2)
            log.info(f"Metrics saved to {metrics_file}")
            
        except ImportError:
            log.error("beir package not installed. Cannot run evaluation.")
        except Exception as e:
            log.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
