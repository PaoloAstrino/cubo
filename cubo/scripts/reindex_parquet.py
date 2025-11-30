"""Reindex chunks parquet to FAISS vector store (generate embeddings and insert)."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.retrieval.retriever import DocumentRetriever
from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Reindex parquet chunks to FAISS vector store")
    parser.add_argument("--parquet", required=True, help="Path to chunks parquet file")
    parser.add_argument("--collection", default=config.get("collection_name", "cubo_documents"))
    parser.add_argument("--model-path", default=config.get("model_path", None))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--replace-collection",
        action="store_true",
        help="Replace vector store by deleting it first",
    )
    parser.add_argument(
        "--wipe-db",
        action="store_true",
        help="Delete the FAISS index folder before reindexing (irrevocable)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.collection:
        config.set("collection_name", args.collection)

    df = pd.read_parquet(Path(args.parquet))
    if df is None or df.shape[0] == 0:
        logger.warning(f"Parquet {args.parquet} has no rows; nothing to reindex")
        return
    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()
    metadatas = df.drop(columns=["text"]).to_dict(orient="records")

    # Load embedding model via EmbeddingGenerator
    if args.model_path:
        config.set("model_path", args.model_path)

    # Optionally wipe FAISS index directory (completely delete and recreate the folder)
    if args.wipe_db:
        index_path = config.get("vector_store_path", "./faiss_index")
        try:
            from shutil import rmtree

            rmtree(index_path)
            logger.info(f"Wiped FAISS index folder {index_path}")
        except Exception as e:
            logger.warning(f"Wipe failed or path not found {index_path}: {e}")

    # Initialize the embedding generator (model is required for DocumentRetriever)
    generator = EmbeddingGenerator(batch_size=args.batch_size)
    # We need the model for the retriever constructor, but retriever uses it for query encoding.
    # Here we use generator for document encoding.
    # Retriever constructor expects a model instance.
    retriever = DocumentRetriever(generator.model)

    if args.replace_collection:
        reset_fn = getattr(retriever.collection, "reset", None)
        if callable(reset_fn):
            reset_fn()
            logger.info(f"Reset vector store collection {args.collection}")
        else:
            logger.warning("Vector store does not support reset(); manual cleanup may be required")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} items with batch size {args.batch_size}")
    embeddings = generator.encode(texts, batch_size=args.batch_size)

    # Insert into collection
    logger.info(f"Adding {len(chunk_ids)} chunks to collection {args.collection}")
    retriever._add_chunks_to_collection(embeddings, texts, metadatas, chunk_ids, "reindex")
    logger.info(f"Inserted {len(chunk_ids)} items into collection {args.collection}")
    # Close the retriever to release resources (threadpool, DB handles)
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


if __name__ == "__main__":
    main()
