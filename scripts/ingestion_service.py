#!/usr/bin/env python3
"""
Background service to monitor ingestion runs and trigger deep ingestion automatically for fast-pass-complete runs.
"""
import time
import argparse
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from src.cubo.ingestion.ingestion_manager import IngestionManager
from src.cubo.storage.metadata_manager import get_metadata_manager
from src.cubo.utils.logger import logger


def main(poll_interval: int = 10):
    manager = IngestionManager()
    metadata = get_metadata_manager()
    logger.info("Starting ingestion service; polling for fast_complete runs...")
    try:
        while True:
            runs = metadata.list_runs_by_status('fast_complete')
            for run in runs:
                run_id = run['id']
                # Check if deep already started
                if run.get('status') == 'fast_complete':
                    folder = run.get('source_folder')
                    logger.info(f"Triggering deep ingestion for run {run_id} folder {folder}")
                    manager.start_deep_pass(folder, run.get('output_parquet'), run_id)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Ingestion service stopping")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Background ingestion service')
    parser.add_argument('--interval', type=int, default=10, help='Polling interval in seconds')
    args = parser.parse_args()
    main(args.interval)
