"""
CLI to query the RAG pipeline and get a response.
"""

import argparse
from pathlib import Path

import pandas as pd

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.processing.generator import create_response_generator
from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.retrieval.retriever import HybridRetriever
from cubo.security.security import security_manager
from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("query", type=str, help="The query to ask.")
    parser.add_argument("--top-k", type=int, default=5, help="The number of documents to retrieve.")
    parser.add_argument(
        "--parquet-path", type=str, help="Path to the Parquet file with chunk data."
    )
    return parser.parse_args()


def _initialize_components(args):
    """Initialize BM25, FAISS, and embedding components."""
    chunks_jsonl = config.get("chunks_jsonl_path", "data/chunks.jsonl")
    bm25_stats = config.get("bm25_stats_path", "data/bm25_stats.json")
    faiss_index_dir = config.get("faiss_index_dir", "faiss_index")
    faiss_index_root = config.get("faiss_index_root", None)

    bm25_searcher = BM25Searcher(chunks_jsonl=chunks_jsonl, bm25_stats=bm25_stats)
    embedding_generator = EmbeddingGenerator()
    
    faiss_manager = FAISSIndexManager(
        dimension=0,
        index_dir=Path(faiss_index_dir),
        index_root=Path(faiss_index_root) if faiss_index_root else None,
    )
    faiss_manager.load()
    
    return bm25_searcher, embedding_generator, faiss_manager


def _initialize_reranker(embedding_generator, top_k):
    """Initialize reranker with fallback options."""
    try:
        reranker_model = config.get("retrieval.reranker_model", None)
        if reranker_model:
            from cubo.rerank.reranker import CrossEncoderReranker
            return CrossEncoderReranker(model_name=reranker_model, top_n=top_k)
    except Exception:
        pass
    
    try:
        from cubo.rerank.reranker import LocalReranker
        return LocalReranker(embedding_generator.model)
    except Exception:
        return None


def _format_context(args, retrieved_docs):
    """Format context from retrieved documents with optional parquet fallback."""
    if args.parquet_path:
        logger.info("Performing retrieval fallback to original chunks...")
        df = pd.read_parquet(args.parquet_path)
        doc_ids = [doc["doc_id"] for doc in retrieved_docs]
        context_df = df[df["chunk_id"].isin(doc_ids)]
        return "\n".join(context_df["text"].tolist())
    return "\n".join([doc["text"] for doc in retrieved_docs])


def _print_results(response, retrieved_docs, parquet_path):
    """Print response and source documents."""
    print("\n--- Response ---")
    print(response)
    print("\n--- Sources ---")
    for doc in retrieved_docs:
        print(f"- {doc['doc_id']} (Score: {doc.get('score', 'N/A'):.4f})")
        if parquet_path:
            print("  (Retrieved from summary, response generated from full text)")


def main():
    args = parse_args()

    bm25_searcher, embedding_generator, faiss_manager = _initialize_components(args)
    reranker = _initialize_reranker(embedding_generator, args.top_k)

    hybrid_retriever = HybridRetriever(
        bm25_searcher=bm25_searcher,
        faiss_manager=faiss_manager,
        embedding_generator=embedding_generator,
        documents=bm25_searcher.docs,
        reranker=reranker,
    )

    response_generator = create_response_generator()

    logger.info(f"Retrieving documents for query: '{security_manager.scrub(args.query)}'")
    retrieved_docs = hybrid_retriever.search(args.query, top_k=args.top_k)

    context = _format_context(args, retrieved_docs)

    logger.info("Generating response...")
    response = response_generator.generate_response(args.query, context)

    _print_results(response, retrieved_docs, args.parquet_path)


if __name__ == "__main__":
    main()
