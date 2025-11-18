"""CLI to build FAISS hot/cold indexes from chunk parquet data."""
from pathlib import Path
import argparse
import logging

import pandas as pd

from src.cubo.config import config
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build hot/cold FAISS indexes from chunk parquet")
    parser.add_argument('--parquet', required=True, help='Parquet file containing chunk text and ids')
    parser.add_argument('--text-column', default='text', help='Name of the text column in the parquet file')
    parser.add_argument('--id-column', default='chunk_id', help='Column containing chunk ids')
    parser.add_argument('--index-dir', default=config.get('faiss_index_dir', 'faiss_index'))
    parser.add_argument('--batch-size', type=int, default=config.get('embedding_batch_size', 32))
    parser.add_argument('--hot-fraction', type=float, default=0.25)
    parser.add_argument('--nlist', type=int, default=64)
    parser.add_argument('--hnsw-m', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true', help='Generate indexes without persisting them')
    parser.add_argument('--verbose', action='store_true', help='Log progress details')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    texts = df[args.text_column].fillna('').astype(str).tolist()
    ids = df[args.id_column].astype(str).tolist()

    if not texts:
        logger.warning("No chunks found in parquet file; nothing to index")
        return
    if len(texts) != len(ids):
        raise ValueError("Text and id columns must have the same length")

    generator = EmbeddingGenerator(batch_size=args.batch_size)
    embeddings = generator.encode(texts, batch_size=args.batch_size)
    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        raise ValueError("Unable to determine embedding dimension")

    manager = FAISSIndexManager(
        dimension=dimension,
        index_dir=Path(args.index_dir),
        nlist=args.nlist,
        hnsw_m=args.hnsw_m,
        hot_fraction=args.hot_fraction
    )
    manager.build_indexes(embeddings, ids)
    sample_query = embeddings[0]
    hits = manager.search(sample_query, k=min(5, len(ids)))
    logger.info(f"Sample search returned {len(hits)} hits (first id: {hits[0]['id'] if hits else 'none'})")

    if not args.dry_run:
        manager.save()
    else:
        logger.info("Dry-run enabled; FAISS indexes were not saved")


if __name__ == '__main__':
    main()