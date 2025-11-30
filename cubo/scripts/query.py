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


def main():
    args = parse_args()

    # 1. Instantiate the components
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

    # Optionally initialize reranker for improved ranking
    reranker = None
    try:
        reranker_model = config.get("retrieval.reranker_model", None)
        if reranker_model:
            from cubo.rerank.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name=reranker_model, top_n=args.top_k)
    except Exception:
        # CrossEncoder unavailable or failed; try LocalReranker
        try:
            from cubo.rerank.reranker import LocalReranker

            reranker = LocalReranker(embedding_generator.model)
        except Exception:
            reranker = None

    hybrid_retriever = HybridRetriever(
        bm25_searcher=bm25_searcher,
        faiss_manager=faiss_manager,
        embedding_generator=embedding_generator,
        documents=bm25_searcher.docs,
        reranker=reranker,
    )

    response_generator = create_response_generator()

    # 2. Retrieve documents
    logger.info(f"Retrieving documents for query: '{security_manager.scrub(args.query)}'")
    retrieved_docs = hybrid_retriever.search(args.query, top_k=args.top_k)

    # 3. Format context with fallback to original chunks
    if args.parquet_path:
        logger.info("Performing retrieval fallback to original chunks...")
        df = pd.read_parquet(args.parquet_path)
        doc_ids = [doc["doc_id"] for doc in retrieved_docs]
        # Assuming the parquet file has a 'chunk_id' and 'text' column
        context_df = df[df["chunk_id"].isin(doc_ids)]
        context = "\n".join(context_df["text"].tolist())
    else:
        context = "\n".join([doc["text"] for doc in retrieved_docs])

    # 4. Generate response
    logger.info("Generating response...")
    response = response_generator.generate_response(args.query, context)

    # 5. Print response and sources
    print("\n--- Response ---")
    print(response)
    print("\n--- Sources ---")
    for doc in retrieved_docs:
        print(f"- {doc['doc_id']} (Score: {doc.get('score', 'N/A'):.4f})")
        if args.parquet_path:
            print("  (Retrieved from summary, response generated from full text)")


if __name__ == "__main__":
    main()
