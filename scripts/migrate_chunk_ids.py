"""Migration tool to migrate ChromaDB chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
import os
from typing import List, Dict

from src.config import config
from src.model_inference_threading import get_model_inference_threading
from src.retriever import DocumentRetriever
from src.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate chunk IDs from filename-based to file-hash-based IDs")
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('chroma_db_path', './chroma_db'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', action='store_true', default=True, help='Prefer file_hash for new chunk ids')
    parser.add_argument('--safe-apply', action='store_true', help='Verify new ids created successfully before deleting old ids')
    return parser.parse_args()


def main():
    args = parse_args()
    # If db path passed, set it in config so retriever uses correct ChromaDB path
    if args.db_path:
        config.set('chroma_db_path', args.db_path)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else None

    planned_changes = []
    # Iterate through all items by index, compute new ids explicitly using file_hash
    for idx, old_id in enumerate(ids):
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        filename = meta.get('filename') or meta.get('filepath') or 'unknown'
        base = meta.get('file_hash') or filename
        logger.debug(f"Index {idx}: old_id={old_id}, file_hash={meta.get('file_hash')}, filename={filename}")
        if meta.get('sentence_index') is not None:
            new_id = f"{base}_s{meta.get('sentence_index')}"
        elif meta.get('page') is not None and meta.get('table_index') is not None:
            new_id = f"{base}_p{meta.get('page')}_t{meta.get('table_index')}"
        elif meta.get('page') is not None:
            if meta.get('sentence_index') is not None:
                new_id = f"{base}_p{meta.get('page')}_s{meta.get('sentence_index')}"
            else:
                new_id = f"{base}_p{meta.get('page')}_chunk_{meta.get('chunk_index', 0)}"
        elif meta.get('chunk_index') is not None:
            new_id = f"{base}_chunk_{meta.get('chunk_index')}"
        else:
            new_id = f"{base}_chunk_{idx}"
        if old_id != new_id:
            planned_changes.append({'old_id': old_id, 'new_id': new_id, 'doc': doc, 'meta': meta, 'embedding': emb})
        else:
            logger.debug(f"No change for {old_id} -> {new_id}")

    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return

    if args.backup:
        backup_path = Path(args.backup)
        with open(backup_path, 'w', encoding='utf-8') as fh:
            for change in planned_changes:
                fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
        logger.info(f"Backed up {len(planned_changes)} metadata entries to {backup_path}")

    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    # Apply changes: add new ids then delete old ids
    # Apply changes: add new ids then delete old ids
    applied = []
    for change in planned_changes:
        try:
            # Add new entry (if embeddings exist, include them)
            ids_to_add = [change['new_id']]
            docs_to_add = [change['doc']]
            metas_to_add = [change['meta']]
            if change['embedding'] is not None:
                coll.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']], metadatas=metas_to_add)
            else:
                coll.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            # Verify new id exists
            verify = coll.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as e:
            logger.error(f"Failed to add new id {change['new_id']}: {e}")
            raise

    # If safe-apply, verify new ids exist before removing old ones
    if args.safe_apply:
        bad = []
        for c in applied:
            res = coll.get(ids=[c['new_id']])
            if not res.get('ids'):
                bad.append(c)
        if bad:
            logger.error("Some new ids could not be verified; aborting deletion")
            logger.error([b['new_id'] for b in bad])
            raise RuntimeError("Safe-apply verification failed")

    # Delete old ids
    # Delete only old ids that were successfully applied
    old_ids = [c['old_id'] for c in applied]
    try:
        coll.delete(ids=old_ids)
        logger.info(f"Deleted {len(old_ids)} old ids after migration")
    except Exception as e:
        logger.error(f"Failed to delete old ids: {e}")
        raise

    logger.info(f"Migration completed: {len(planned_changes)} entries changed")


if __name__ == '__main__':
    main()
