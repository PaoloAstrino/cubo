#!/usr/bin/env python3
"""
Script to run fast pass ingestion for immediate BM25 availability.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.fast_pass_ingestor import build_bm25_index
from src.ingest.deep_ingestor import DeepIngestor
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Fast pass ingestion for CUBO")
    parser.add_argument('folder', help='Folder to ingest')
    parser.add_argument('--output', default=None, help='Output directory for fast pass')
    parser.add_argument('--skip-model', action='store_true', help='Skip loading heavy models (Dolphin/Embeddings)')
    parser.add_argument('--deep', action='store_true', help='Run deep ingestion after fast pass')
    args = parser.parse_args()

    result = build_bm25_index(args.folder, args.output, skip_model=args.skip_model)
    if result:
        logger.info(f"Fast pass ingest completed. Chunks: {result['chunks_count']}")
        if 'chunks_parquet' in result:
            logger.info(f"Chunks parquet: {result['chunks_parquet']}")
        if 'chunks_jsonl' in result:
            logger.info(f"Chunks jsonl: {result['chunks_jsonl']}")
        logger.info(f"BM25 stats: {result['bm25_stats']}")
        if args.deep:
            deep_out = args.output or None
            dee = DeepIngestor(input_folder=args.folder, output_dir=deep_out)
            deep_result = dee.ingest()
            if deep_result:
                logger.info(f"Deep ingest completed. Chunks: {deep_result['chunks_count']}")
                logger.info(f"Deep chunks parquet: {deep_result['chunks_parquet']}")
            else:
                logger.warning('Deep ingest produced no chunks')
    else:
        logger.error('Fast pass ingest did not produce results')


if __name__ == '__main__':
    main()
