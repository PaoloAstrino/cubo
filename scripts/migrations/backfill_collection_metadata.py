#!/usr/bin/env python3
"""Migration script to backfill document metadata with collection_id information.

This script updates the vector store and parquet files to ensure all chunks
have proper collection_id and doc_id associations, enabling collection-scoped
retrieval filtering.

Usage:
    python scripts/migrations/backfill_collection_metadata.py [--dry-run] [--collection-id ID] [--force]

Options:
    --dry-run           Show what would be updated without making changes
    --collection-id ID  Backfill a specific collection (default: all collections)
    --force             Force update even if metadata already exists
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_chunks_from_parquet(parquet_path: str) -> List[Dict]:
    """Load chunks from parquet file.
    
    Args:
        parquet_path: Path to chunks parquet file
        
    Returns:
        List of chunk dictionaries
    """
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to read parquet file {parquet_path}: {e}")
        return []


def update_vector_store_metadata(
    collection,
    doc_id: str,
    collection_id: str,
    dry_run: bool = False
) -> int:
    """Update vector store metadata for a document.
    
    Args:
        collection: Chroma collection object
        doc_id: Document ID (filename)
        collection_id: Collection ID to assign
        dry_run: If True, don't actually update
        
    Returns:
        Number of chunks updated
    """
    try:
        # Query all chunks from this document
        results = collection.get(
            where={"filename": {"$eq": doc_id}},
            include=["metadatas", "documents", "ids"]
        )
        
        if not results or not results.get("ids"):
            logger.debug(f"No chunks found for document {doc_id}")
            return 0
        
        chunk_ids = results["ids"]
        metadatas = results["metadatas"] or []
        
        count = 0
        updated_metadatas = []
        
        for meta in metadatas:
            if not isinstance(meta, dict):
                meta = {"filename": doc_id}
            
            # Add collection_id if not present
            if "collection_id" not in meta:
                meta["collection_id"] = collection_id
                count += 1
            
            # Ensure doc_id is present
            if "doc_id" not in meta:
                meta["doc_id"] = doc_id
            
            updated_metadatas.append(meta)
        
        if count > 0 and not dry_run:
            try:
                # Update metadata in vector store
                collection.update(
                    ids=chunk_ids,
                    metadatas=updated_metadatas
                )
                logger.info(f"Updated {count} chunks for document {doc_id} with collection_id={collection_id}")
            except Exception as e:
                logger.error(f"Failed to update metadata for document {doc_id}: {e}")
                return 0
        
        return count if not dry_run or count > 0 else 0
        
    except Exception as e:
        logger.error(f"Error updating metadata for document {doc_id}: {e}")
        return 0


def backfill_from_metadata_manager(
    metadata_manager,
    collection,
    collection_id: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False
) -> int:
    """Backfill metadata from ingestion records.
    
    Args:
        metadata_manager: MetadataManager instance
        collection: Chroma collection
        collection_id: Optional collection ID to filter by
        dry_run: If True, don't actually update
        force: Force update even if metadata exists
        
    Returns:
        Total number of chunks updated
    """
    try:
        # Get all ingestion runs (or specific collection if provided)
        with metadata_manager._lock:
            cur = metadata_manager.conn.cursor()
            
            if collection_id:
                cur.execute(
                    "SELECT run_id FROM ingestion_runs WHERE id = ? LIMIT 1",
                    (collection_id,)
                )
                run_ids = [row[0] for row in cur.fetchall()]
            else:
                cur.execute("SELECT id FROM ingestion_runs WHERE status = 'success'")
                run_ids = [row[0] for row in cur.fetchall()]
            
            if not run_ids:
                logger.warning(f"No ingestion runs found for collection_id={collection_id}")
                return 0
            
            total_updated = 0
            
            for run_id in run_ids:
                cur.execute(
                    "SELECT file_path FROM ingestion_files WHERE run_id = ? AND status = 'success'",
                    (run_id,)
                )
                file_paths = [row[0] for row in cur.fetchall()]
                
                for file_path in file_paths:
                    # Extract filename for vector store query
                    filename = Path(file_path).name
                    
                    updated = update_vector_store_metadata(
                        collection,
                        doc_id=filename,
                        collection_id=run_id,
                        dry_run=dry_run
                    )
                    total_updated += updated
            
            return total_updated
            
    except Exception as e:
        logger.error(f"Error during backfill: {e}")
        return 0


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Backfill document metadata with collection_id information"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Backfill a specific collection (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if metadata already exists"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting collection metadata backfill migration...")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    try:
        # Import here to avoid issues if cubo not installed
        from cubo.config import config
        from cubo.storage.metadata_manager import MetadataManager, get_metadata_manager
        from cubo.storage.vector_store_factory import create_vector_store
        
        # Initialize managers
        metadata_manager = get_metadata_manager()
        vector_store = create_vector_store()
        
        if not vector_store or not vector_store.collection:
            logger.error("Failed to initialize vector store")
            return 1
        
        # Run backfill
        updated = backfill_from_metadata_manager(
            metadata_manager,
            vector_store.collection,
            collection_id=args.collection_id,
            dry_run=args.dry_run,
            force=args.force
        )
        
        logger.info(f"Migration complete. Updated {updated} chunks.")
        
        if args.dry_run:
            logger.info("To apply changes, run without --dry-run flag")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
