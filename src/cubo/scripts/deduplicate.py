"""
CLI to run the deduplication pipeline on a corpus of documents.
"""
import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from src.cubo.deduplication.deduplicator import Deduplicator
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run the deduplication pipeline.")
    parser.add_argument('--input-parquet', required=True, help='Path to the Parquet file with the documents.')
    parser.add_argument('--output-map', required=True, help='Path to save the deduplication map.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Jaccard similarity threshold.')
    parser.add_argument('--num-perm', type=int, default=128, help='Number of permutations for MinHash.')
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load documents
    logger.info(f"Loading documents from {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    # The Deduplicator expects a list of dictionaries with 'doc_id' and 'text' keys.
    # Let's assume the Parquet file has 'chunk_id' and 'text' columns.
    documents = df.rename(columns={'chunk_id': 'doc_id'}).to_dict('records')

    # 2. Instantiate the Deduplicator
    deduplicator = Deduplicator(threshold=args.threshold, num_perm=args.num_perm)

    # 3. Run the deduplication process
    logger.info("Running deduplication...")
    canonical_map = deduplicator.deduplicate(documents)

    # 4. Save the deduplication map
    output_path = Path(args.output_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'version': '1.0',
        'created_at': datetime.datetime.utcnow().isoformat(),
        'threshold': args.threshold,
        'num_perm': args.num_perm,
        'canonical_map': canonical_map,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Deduplication map saved to {output_path}")
    logger.info(f"Found {len(set(canonical_map.values()))} canonical documents out of {len(documents)} total documents.")

if __name__ == '__main__':
    main()
