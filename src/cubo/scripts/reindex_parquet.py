"""Reindex chunks parquet to ChromaDB collection (generate embeddings and insert)."""
from pathlib import Path
import argparse
import logging
import pandas as pd

from src.cubo.embeddings.model_loader import model_manager
from tqdm import tqdm
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.config import config
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Reindex parquet chunks to ChromaDB")
    parser.add_argument('--parquet', required=True, help='Path to chunks parquet file')
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--model-path', default=config.get('model_path', None))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--replace-collection', action='store_true', help='Replace collection by deleting it first')
    parser.add_argument('--wipe-db', action='store_true', help='Delete the whole chroma DB folder before reindexing (irrevocable)')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    df = pd.read_parquet(Path(args.parquet))
    texts = df['text'].tolist()
    chunk_ids = df['chunk_id'].tolist()
    metadatas = df.drop(columns=['text']).to_dict(orient='records')

    # Load embedding model
    # Load embedding model via model manager
    if args.model_path:
        config.set('model_path', args.model_path)
    model = model_manager.get_model()
    retriever = DocumentRetriever(model)
    # Optionally wipe DB directory (completely delete and recreate the folder)
    if args.wipe_db:
        db_path = config.get('chroma_db_path')
        try:
            from shutil import rmtree
            rmtree(db_path)
            logger.info(f"Wiped chroma db folder {db_path}")
        except Exception as e:
            logger.warning(f"Wipe failed or path not found {db_path}: {e}")

    if args.replace_collection:
        try:
            retriever.client.delete_collection(args.collection)
            logger.info(f"Deleted collection {args.collection} to replace it")
        except Exception:
            # ignore if delete fails (collection not present)
            pass

    retriever.collection = retriever.client.get_or_create_collection(args.collection)

    # Generate embeddings with progress
    batch_size = args.batch_size
    embeddings = []
    logger.info(f"Generating embeddings for {len(texts)} items with batch size {batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb_batch = retriever.inference_threading.generate_embeddings_threaded(batch, model, batch_size=batch_size)
        embeddings.extend(emb_batch)
        logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}")

    # Insert into collection
    # Insert into collection with progress
    logger.info(f"Adding {len(chunk_ids)} chunks to collection {args.collection}")
    retriever._add_chunks_to_collection(embeddings, texts, metadatas, chunk_ids, 'reindex')
    logger.info(f"Inserted {len(chunk_ids)} items into collection {args.collection}")


if __name__ == '__main__':
    main()
