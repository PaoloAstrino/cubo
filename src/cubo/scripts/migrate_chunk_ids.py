#!/usr/bin/env python3
"""Migration tool to migrate chunk IDs from filename-based to file-hash-based IDs.

Usage:
    python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
    python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
        parser = argparse.ArgumentParser(
                description="Migrate chunk IDs from filename-based to file-hash-based IDs"
        )
        parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
        parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--apply', action='store_true')
        parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
        parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                                                help='Prefer file_hash for new chunk ids')
        parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                                                help='Do not prefer file_hash when generating new chunk ids')
        parser.add_argument('--safe-apply', action='store_true',
                                                help='Verify new ids created successfully before deleting old ids')
        parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
        return parser.parse_args()
#!/usr/bin/env python3
"""Migration tool to migrate chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, Any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            if hasattr(coll, 'delete'):
                coll.delete(ids=old_ids)
                logger.info(f"Deleted {len(old_ids)} old ids after migration")
            else:
                logger.warning("Vector store backend does not support direct deletion; manual removal required")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Migration tool to migrate chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, Any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            if hasattr(coll, 'delete'):
                coll.delete(ids=old_ids)
                logger.info(f"Deleted {len(old_ids)} old ids after migration")
            else:
                logger.warning("Vector store backend does not support direct deletion; manual removal required")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Migration tool to migrate chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, Any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            if hasattr(coll, 'delete'):
                coll.delete(ids=old_ids)
                logger.info(f"Deleted {len(old_ids)} old ids after migration")
            else:
                logger.warning("Vector store backend does not support direct deletion; manual removal required")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
    parser.add_argument('--safe-apply', action='store_true', help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = []
    # Iterate through all items by index and compute new ids using file_hash preference
    for idx, old_id in enumerate(ids):
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        base = meta.get('file_hash') or meta.get('filename') or meta.get('filepath') or 'unknown'
        if meta.get('sentence_index') is not None:
            new_id = f"{base}_s{meta.get('sentence_index')}"
        elif meta.get('page') is not None and meta.get('table_index') is not None:
            new_id = f"{base}_p{meta.get('page')}_t{meta.get('table_index')}"
        elif meta.get('page') is not None:
            new_id = f"{base}_p{meta.get('page')}_chunk_{meta.get('chunk_index', idx)}"
        elif meta.get('chunk_index') is not None:
            new_id = f"{base}_chunk_{meta.get('chunk_index')}"
        else:
            new_id = f"{base}_chunk_{idx}"
        if old_id != new_id:
            planned_changes.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })

    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        # backup metadata as JSONL
        backup_file = Path(args.backup)
        with open(backup_file, 'w', encoding='utf-8') as fh:
            for change in planned_changes:
                fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
        logger.info(f"Backed up metadata to {backup_file}")
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    # Apply changes
    applied = []
    for change in (tqdm(planned_changes) if args.verbose else planned_changes):
        try:
            if change['embedding'] is not None:
                coll.add(ids=[change['new_id']], documents=[change['doc']], embeddings=[change['embedding']], metadatas=[change['meta']])
            else:
                coll.add(ids=[change['new_id']], documents=[change['doc']], metadatas=[change['meta']])
            if coll.get(ids=[change['new_id']]).get('ids'):
                applied.append(change)
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add; aborting")
                raise RuntimeError("Verification failed")
        except Exception as e:
            logger.error(f"Failed to apply change for {change['old_id']} -> {change['new_id']}: {e}")
            raise

    if args.safe_apply:
        bad = [c for c in applied if not coll.get(ids=[c['new_id']]).get('ids')]
        if bad:
            logger.error("Safe-apply verification failed; aborting deletion of old ids")
            raise RuntimeError("Safe-apply verification failed")

    # delete old ids
    try:
        old_ids = [c['old_id'] for c in applied]
        coll.delete(ids=old_ids)
        logger.info(f"Deleted {len(old_ids)} old ids after migration")
    except Exception as e:
        logger.error(f"Failed to delete old ids: {e}")
        raise

    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Migration tool to migrate vector store chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, Any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            coll.delete(ids=old_ids)
            logger.info(f"Deleted {len(old_ids)} old ids after migration")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Migration tool to migrate vector store chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List
import os

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                logger.error(f"Failed to verify new id {change['new_id']} after add")
                raise RuntimeError(f"Verification failed for {change['new_id']}")
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, Any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else []

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            coll.delete(ids=old_ids)
            logger.info(f"Deleted {len(old_ids)} old ids after migration")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()
"""Migration tool to migrate vector store chunk IDs from filename-based to file-hash-based IDs.

Usage:
  python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
  python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
from typing import Any, Dict, List

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate chunk IDs from filename-based to file-hash-based IDs"
    )
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', dest='chunk_id_use_file_hash', action='store_true', default=True,
                        help='Prefer file_hash for new chunk ids')
    parser.add_argument('--no-chunk-id-file-hash', dest='chunk_id_use_file_hash', action='store_false',
                        help='Do not prefer file_hash when generating new chunk ids')
    parser.add_argument('--safe-apply', action='store_true',
                        help='Verify new ids created successfully before deleting old ids')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars and detailed logging')
    return parser.parse_args()


def _compute_base(meta: Dict[str, Any], use_file_hash: bool) -> str:
    if use_file_hash and meta.get('file_hash'):
        return meta.get('file_hash')
    return meta.get('filename') or meta.get('filepath') or 'unknown'


def _generate_new_id(meta: Dict[str, Any], idx: int, use_file_hash: bool) -> str:
    base = _compute_base(meta, use_file_hash)
    sentence_index = meta.get('sentence_index')
    page = meta.get('page')
    table_index = meta.get('table_index')
    chunk_index = meta.get('chunk_index')

    if sentence_index is not None and page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}_s{sentence_index}"
    if sentence_index is not None and page is not None:
        return f"{base}_p{page}_s{sentence_index}"
    if sentence_index is not None:
        return f"{base}_s{sentence_index}"
    if page is not None and table_index is not None:
        return f"{base}_p{page}_t{table_index}"
    if page is not None:
        return f"{base}_p{page}_chunk_{chunk_index or idx}"
    if chunk_index is not None:
        return f"{base}_chunk_{chunk_index}"
    return f"{base}_chunk_{idx}"


def _plan_changes(ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str],
                  embeddings: List[List[float]], use_file_hash: bool, verbose: bool) -> List[Dict[str, Any]]:
    planned = []
    iterator = enumerate(ids)
    if verbose:
        iterator = tqdm(iterator, total=len(ids), desc="Scanning chunks")
    for idx, old_id in iterator:
        meta = metadatas[idx]
        doc = documents[idx]
        emb = embeddings[idx] if embeddings else None
        new_id = _generate_new_id(meta, idx, use_file_hash)
        if old_id != new_id:
            planned.append({
                'old_id': old_id,
                'new_id': new_id,
                'doc': doc,
                'meta': meta,
                'embedding': emb
            })
            logger.debug(f"Planned change {old_id} -> {new_id}")
        else:
            logger.debug(f"No change for {old_id}")
    return planned


def _backup_metadata(changes: List[Dict[str, Any]], backup_path: Path) -> None:
    with open(backup_path, 'w', encoding='utf-8') as fh:
        for change in changes:
            fh.write(json.dumps(change['meta'], ensure_ascii=False) + '\n')
    logger.info(f"Backed up {len(changes)} metadata entries to {backup_path}")


def _apply_changes(collection, changes: List[Dict[str, Any]], verbose: bool) -> List[Dict[str, Any]]:
    applied = []
    iterator = changes
    if verbose:
        iterator = tqdm(changes, desc="Applying migrations")
    for change in iterator:
        ids_to_add = [change['new_id']]
        docs_to_add = [change['doc']]
        metas_to_add = [change['meta']]
        try:
            if change['embedding'] is not None:
                collection.add(ids=ids_to_add, documents=docs_to_add, embeddings=[change['embedding']],
                               metadatas=metas_to_add)
            else:
                collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)
            verify = collection.get(ids=ids_to_add)
            if verify.get('ids'):
                applied.append(change)
                logger.info(f"Added new id {change['new_id']} successfully")
            else:
                if __name__ == '__main__':
                    main()
        except Exception as exc:
            logger.error(f"Failed to add new id {change['new_id']}: {exc}")
            raise
    return applied


def _safe_verify(collection, applied: List[Dict[str, any]]) -> None:
    bad = [change for change in applied if not collection.get(ids=[change['new_id']]).get('ids')]
    if bad:
        logger.error("Safe-apply verification failed for the following ids:")
        for change in bad:
            logger.error(f"Missing {change['new_id']}")
        raise RuntimeError("Safe-apply verification failed")


def main():
    args = parse_args()
    if args.db_path:
        config.set('vector_store_path', args.db_path)
    config.set('collection_name', args.collection)
    retriever = DocumentRetriever(model=None)
    coll = retriever.collection

    all_data = coll.get()
    ids = all_data.get('ids', [])
    metadatas = all_data.get('metadatas', [])
    documents = all_data.get('documents', [])
    embeddings = all_data.get('embeddings', []) if 'embeddings' in all_data else None

    planned_changes = _plan_changes(ids, metadatas, documents, embeddings,
                                    args.chunk_id_use_file_hash, args.verbose)
    logger.info(f"Found {len(planned_changes)} ids that will change")
    if not planned_changes:
        logger.info("No planned changes detected. Nothing to migrate.")
        return
    if args.dry_run:
        for change in planned_changes:
            print(f"Would change {change['old_id']} -> {change['new_id']}")
        return
    if args.backup:
        _backup_metadata(planned_changes, Path(args.backup))
    if not args.apply:
        logger.info("No changes applied. Run with --apply to actually execute the migration.")
        return

    applied = _apply_changes(coll, planned_changes, args.verbose)
    if args.safe_apply:
        _safe_verify(coll, applied)

    old_ids = [change['old_id'] for change in applied]
    if old_ids:
        try:
            coll.delete(ids=old_ids)
            logger.info(f"Deleted {len(old_ids)} old ids after migration")
        except Exception as exc:
            logger.error(f"Failed to delete old ids: {exc}")
            raise
    logger.info(f"Migration completed: {len(applied)} entries changed")


if __name__ == '__main__':
    main()"""Migration tool to migrate vector store chunk IDs from filename-based to file-hash-based IDs.

Usage:
    python scripts/migrate_chunk_ids.py --collection cubo_documents --dry-run
    python scripts/migrate_chunk_ids.py --collection cubo_documents --apply --backup backup.jsonl
"""
from pathlib import Path
import argparse
import json
import os
from typing import List, Dict

from tqdm import tqdm
from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate chunk IDs from filename-based to file-hash-based IDs")
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=config.get('vector_store_path', './faiss_index'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', default=None, help='Path to JSONL backup before applying changes')
    parser.add_argument('--chunk-id-use-file-hash', action='store_true', default=True, help='Prefer file_hash for new chunk ids')
    parser.add_argument('--safe-apply', action='store_true', help='Verify new ids created successfully before deleting old ids')
    return parser.parse_args()


def main():
    args = parse_args()
    # If db path passed, set it in config so retriever uses correct vector store path
    if args.db_path:
        config.set('vector_store_path', args.db_path)
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
