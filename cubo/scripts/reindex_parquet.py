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


def _load_parquet_data(parquet_path):
    """Load and validate parquet data."""
    df = pd.read_parquet(Path(parquet_path))
    if df is None or df.shape[0] == 0:
        logger.warning(f"Parquet {parquet_path} has no rows; nothing to reindex")
        return None, None, None
    
    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()
    metadatas = df.drop(columns=["text"]).to_dict(orient="records")
    return texts, chunk_ids, metadatas


def _wipe_index_if_requested(args):
    """Wipe FAISS index directory if requested."""
    if not args.wipe_db:
        return
    
    index_path = config.get("vector_store_path", "./faiss_index")
    try:
        from shutil import rmtree
        rmtree(index_path)
        logger.info(f"Wiped FAISS index folder {index_path}")
    except Exception as e:
        logger.warning(f"Wipe failed or path not found {index_path}: {e}")


def _initialize_generator_and_retriever(args):
    """Initialize embedding generator and retriever."""
    generator = EmbeddingGenerator(batch_size=args.batch_size)
    retriever = DocumentRetriever(generator.model)
    return generator, retriever


def _reset_collection_if_requested(args, retriever):
    """Reset collection if requested."""
    if not args.replace_collection:
        return
    
    reset_fn = getattr(retriever.collection, "reset", None)
    if callable(reset_fn):
        reset_fn()
        logger.info(f"Reset vector store collection {args.collection}")
    else:
        logger.warning("Vector store does not support reset(); manual cleanup may be required")


def _generate_and_insert_embeddings(generator, retriever, texts, chunk_ids, metadatas, args):
    """Generate embeddings and insert into collection."""
    logger.info(
        f"Generating embeddings for {len(texts)} items with batch size {args.batch_size} (document prompt)"
    )
    embeddings = generator.encode(texts, batch_size=args.batch_size, prompt_name="document")

    logger.info(f"Adding {len(chunk_ids)} chunks to collection {args.collection}")
    retriever._add_chunks_to_collection(embeddings, texts, metadatas, chunk_ids, "reindex")
    logger.info(f"Inserted {len(chunk_ids)} items into collection {args.collection}")


def _cleanup_retriever(retriever):
    """Close retriever to release resources."""
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.collection:
        config.set("collection_name", args.collection)

    texts, chunk_ids, metadatas = _load_parquet_data(args.parquet)
    if texts is None:
        return

    if args.model_path:
        config.set("model_path", args.model_path)

    _wipe_index_if_requested(args)
    generator, retriever = _initialize_generator_and_retriever(args)
    _reset_collection_if_requested(args, retriever)
    _generate_and_insert_embeddings(generator, retriever, texts, chunk_ids, metadatas, args)
    _cleanup_retriever(retriever)


if __name__ == "__main__":
    main()
