#!/usr/bin/env python3
"""
Utility to sanitize existing chunk IDs with spaces and special characters.

This script fixes the "Invalid id value detected" error by renaming chunk IDs
that contain spaces or special characters not allowed by the ID_RE regex pattern.

Spaces and special chars are replaced with underscores.

Usage:
    python scripts/sanitize_chunk_ids.py --collection cubo_documents --dry-run
    python scripts/sanitize_chunk_ids.py --collection cubo_documents --apply
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever
from cubo.utils.logger import logger

# Pattern for valid IDs (must match vector_store.py ID_RE)
VALID_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
# Pattern to sanitize invalid characters
SANITIZE_RE = re.compile(r"[^A-Za-z0-9._:-]")


def sanitize_id(old_id: str) -> Optional[str]:
    """Sanitize an ID by replacing invalid characters with underscores.

    Returns None if ID is already valid, otherwise returns sanitized version.
    """
    if VALID_ID_RE.match(old_id):
        return None  # Already valid
    return SANITIZE_RE.sub("_", old_id)


def scan_collection_for_invalid_ids(
    collection: Any,
) -> List[Dict[str, Any]]:
    """Scan collection for IDs with invalid characters.

    Returns list of dicts with 'old_id', 'new_id', and 'valid' status.
    """
    all_data = collection.get()
    ids = all_data.get("ids", [])
    metadatas = all_data.get("metadatas", [])
    documents = all_data.get("documents", [])
    embeddings = all_data.get("embeddings", []) if "embeddings" in all_data else []

    invalid_ids = []
    for idx, old_id in enumerate(ids):
        new_id = sanitize_id(old_id)
        if new_id is not None:
            invalid_ids.append(
                {
                    "old_id": old_id,
                    "new_id": new_id,
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "document": documents[idx] if idx < len(documents) else "",
                    "embedding": embeddings[idx] if idx < len(embeddings) else None,
                }
            )

    return invalid_ids


def apply_sanitization(
    collection: Any,
    changes: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Apply ID sanitization changes to the collection.

    Returns list of successfully applied changes.
    """
    applied = []

    for change in changes:
        try:
            old_id = change["old_id"]
            new_id = change["new_id"]

            # Get the document data
            result = collection.get(ids=[old_id])
            if not result["ids"]:
                logger.warning(f"ID not found: {old_id}")
                continue

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            document = result["documents"][0] if result["documents"] else ""
            embedding = result["embeddings"][0] if result["embeddings"] else None

            # Delete old ID
            collection.delete(ids=[old_id])

            # Add with new ID
            if embedding is not None:
                collection.upsert(
                    ids=[new_id],
                    metadatas=[metadata],
                    documents=[document],
                    embeddings=[embedding],
                )
            else:
                collection.upsert(
                    ids=[new_id],
                    metadatas=[metadata],
                    documents=[document],
                )

            if verbose:
                logger.info(f"Migrated: {old_id} -> {new_id}")

            applied.append(
                {
                    "old_id": old_id,
                    "new_id": new_id,
                    "status": "success",
                }
            )

        except Exception as e:
            logger.error(f"Failed to migrate {change['old_id']}: {e}")
            applied.append(
                {
                    "old_id": change["old_id"],
                    "new_id": change["new_id"],
                    "status": "failed",
                    "error": str(e),
                }
            )

    return applied


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize chunk IDs with invalid characters (spaces, special chars)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="cubo_documents",
        help="Collection name to sanitize",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to vector store database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to collection",
    )
    parser.add_argument(
        "--backup",
        type=str,
        help="Path to save backup of changes",
    )

    args = parser.parse_args()

    # Configure
    if args.db_path:
        config.set("vector_store_path", args.db_path)

    config.set("collection_name", args.collection)

    # Initialize retriever and collection
    retriever = DocumentRetriever(model=None)
    collection = retriever.collection

    logger.info(f"Scanning collection '{args.collection}' for invalid IDs...")
    invalid_ids = scan_collection_for_invalid_ids(collection)

    if not invalid_ids:
        logger.info("✓ No invalid IDs found. Collection is clean.")
        return

    logger.info(f"Found {len(invalid_ids)} IDs with invalid characters:")
    for item in invalid_ids[:5]:  # Show first 5
        logger.info(f"  {item['old_id']} -> {item['new_id']}")
    if len(invalid_ids) > 5:
        logger.info(f"  ... and {len(invalid_ids) - 5} more")

    if args.dry_run:
        logger.info("\n[DRY RUN] No changes applied. Use --apply to fix.")
        return

    if not args.apply:
        logger.info("\nTo apply changes, run with --apply flag")
        return

    # Backup metadata if requested
    if args.backup:
        backup_path = Path(args.backup)
        logger.info(f"Creating backup at {backup_path}...")
        with open(backup_path, "w") as f:
            for item in invalid_ids:
                f.write(json.dumps(item) + "\n")
        logger.info(f"✓ Backup created")

    # Apply changes
    logger.info("Applying sanitization...")
    applied = apply_sanitization(collection, invalid_ids, verbose=True)

    successful = sum(1 for a in applied if a["status"] == "success")
    failed = sum(1 for a in applied if a["status"] == "failed")

    logger.info(f"\n✓ Sanitization complete: {successful} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
