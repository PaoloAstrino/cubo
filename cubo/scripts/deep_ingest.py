"""CLI script to run deep ingestion for a folder and persist chunks parquet.

Usage:
  python scripts/deep_ingest.py --input data/docs --output data/deep
"""

import argparse
import logging

from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepIngestor over a folder")
    parser.add_argument("--input", required=True, help="Input folder to process")
    parser.add_argument(
        "--output", required=False, help="Output directory for parquet and manifest"
    )
    parser.add_argument(
        "--csv-rows", type=int, default=None, help="Rows per CSV chunk (overrides config)"
    )
    parser.add_argument(
        "--no-filehash",
        dest="filehash",
        action="store_false",
        help="Disable file-hash based chunk IDs (use filename-based IDs)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)

    ingestor = DeepIngestor(
        input_folder=args.input,
        output_dir=args.output,
        csv_rows_per_chunk=args.csv_rows,
        use_file_hash_for_chunk_id=args.filehash,
    )

    result = ingestor.ingest()
    if not result:
        print("No chunks produced")
        return

    print(f"Saved {result['chunks_count']} chunks to: {result['chunks_parquet']}")
    print(f"Manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
