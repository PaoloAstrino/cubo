#!/usr/bin/env python3
"""
Script to run fast pass ingestion for immediate BM25 availability.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cubo.ingestion.ingestion_manager import IngestionManager
from src.cubo.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Fast pass ingestion for CUBO")
    parser.add_argument("folder", help="Folder to ingest")
    parser.add_argument("--output", default=None, help="Output directory for fast pass")
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip loading heavy models (Dolphin/Embeddings)"
    )
    parser.add_argument("--deep", action="store_true", help="Run deep ingestion after fast pass")
    parser.add_argument(
        "--background", action="store_true", help="Run deep ingestion in the background (async)"
    )
    args = parser.parse_args()

    manager = IngestionManager()
    result = manager.start_fast_pass(
        args.folder,
        args.output,
        skip_model=args.skip_model,
        auto_deep=(args.deep or args.background),
    )
    if result:
        logger.info(f"Fast pass ingest completed. Chunks: {result['chunks_count']}")
        if "chunks_parquet" in result:
            logger.info(f"Chunks parquet: {result['chunks_parquet']}")
        if "chunks_jsonl" in result:
            logger.info(f"Chunks jsonl: {result['chunks_jsonl']}")
        logger.info(f"BM25 stats: {result['bm25_stats']}")
        if args.background:
            logger.info("Deep ingestion scheduled in background")
    else:
        logger.error("Fast pass ingest did not produce results")


if __name__ == "__main__":
    main()
