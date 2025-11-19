"""
CLI to query the RAG pipeline and get a response.
"""
import argparse
from pathlib import Path
import pandas as pd

from src.cubo.retrieval.retriever import HybridRetriever
from src.cubo.processing.generator import ResponseGenerator
from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
from src.cubo.config import config
from src.cubo.utils.logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument('query', type=str, help='The query to ask.')
    parser.add_argument('--top-k', type=int, default=5, help='The number of documents to retrieve.')
    parser.add_argument('--parquet-path', type=str, help='Path to the Parquet file with chunk data.')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Instantiate the components
    chunks_jsonl = config.get('chunks_jsonl_path', 'data/chunks.jsonl')
    bm25_stats = config.get('bm25_stats_path', 'data/bm25_stats.json')
    faiss_index_dir = config.get('faiss_index_dir', 'faiss_index')
    
    bm25_searcher = BM25Searcher(chunks_jsonl=chunks_jsonl, bm25_stats=bm25_stats)
    
    embedding_generator = EmbeddingGenerator()
    faiss_manager = FAISSIndexManager(dimension=0, index_dir=Path(faiss_index_dir))
    faiss_manager.load()

    hybrid_retriever = HybridRetriever(
        bm25_searcher=bm25_searcher,
        faiss_manager=faiss_manager,
        embedding_generator=embedding_generator,
        documents=bm25_searcher.docs,
    )

    response_generator = ResponseGenerator()

    # 2. Retrieve documents
    logger.info(f"Retrieving documents for query: '{args.query}'")
    retrieved_docs = hybrid_retriever.search(args.query, top_k=args.top_k)

    # 3. Format context with fallback to original chunks
    if args.parquet_path:
        logger.info("Performing retrieval fallback to original chunks...")
        df = pd.read_parquet(args.parquet_path)
        doc_ids = [doc['doc_id'] for doc in retrieved_docs]
        # Assuming the parquet file has a 'chunk_id' and 'text' column
        context_df = df[df['chunk_id'].isin(doc_ids)]
        context = "\n".join(context_df['text'].tolist())
    else:
        context = "\n".join([doc['text'] for doc in retrieved_docs])

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

if __name__ == '__main__':
    main()
