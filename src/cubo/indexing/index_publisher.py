from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict

import faiss

# Import FAISSIndexManager inside functions to avoid circular import with faiss_index
from src.cubo.storage.metadata_manager import get_metadata_manager
from src.cubo.indexing.publish_lock import acquire_publish_lock
from src.cubo.utils.logger import logger


POINTER_FILENAME = 'current_index.json'


def _verify_index_dir(dir_path: Path) -> Dict[str, Optional[str]]:
    """Verify that the FAISS index directory is loadable.
    Returns metadata dict read from metadata.json if successful.
    Raises Exception on failure.
    """
    meta_path = dir_path / 'metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json missing in {dir_path}")
    with open(meta_path, 'r', encoding='utf-8') as fh:
        metadata = json.load(fh)

    dimension = metadata.get('dimension')
    if dimension is None:
        raise ValueError('Missing dimension in metadata')

    # Try to read indexes
    hot_path = dir_path / 'hot.index'
    cold_path = dir_path / 'cold.index'
    # If they exist, try to read via faiss
    try:
        # If metadata references hot_ids/cold_ids, ensure index files exist and are readable
        hot_ids = metadata.get('hot_ids', []) or []
        cold_ids = metadata.get('cold_ids', []) or []

        if hot_ids and not hot_path.exists():
            raise RuntimeError("metadata claims hot_ids but hot.index is missing")
        if cold_ids and not cold_path.exists():
            raise RuntimeError("metadata claims cold_ids but cold.index is missing")

        if hot_path.exists():
            faiss.read_index(str(hot_path))
        if cold_path.exists():
            faiss.read_index(str(cold_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read index artifacts: {exc}")

    # We can attempt to load into a local FAISSIndexManager and sample a search
    try:
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        manager = FAISSIndexManager(dimension=dimension, index_dir=dir_path)
        manager.load()
        # Run a minimal sanity check if any index has entries
        sample_vec = None
        ntotal_hot = 0
        ntotal_cold = 0
        try:
            ntotal_hot = manager.hot_index.ntotal if manager.hot_index is not None else 0
        except Exception:
            ntotal_hot = 0
        try:
            ntotal_cold = manager.cold_index.ntotal if manager.cold_index is not None else 0
        except Exception:
            ntotal_cold = 0
        if (ntotal_hot + ntotal_cold) > 0:
            # Construct a simple zero vector to attempt a search and ensure query path works
            import numpy as _np
            sample_vec = _np.zeros((manager.dimension,), dtype='float32')
            # search top 1
            results = manager.search(sample_vec.tolist(), k=1)
            # If metadata claims presence (ids) but search returns empty, fail
            if (ntotal_hot + ntotal_cold) > 0 and len(results) == 0:
                raise RuntimeError('Index loaded but sanity search returned zero results')
    except Exception as exc:
        raise RuntimeError(f"Failed to load indexes into FAISSIndexManager: {exc}")

    return metadata


def publish_version(version_dir: Path, index_root: Path, verify: bool = True, version_id: Optional[str] = None) -> Path:
    """Publish a FAISS version directory by verifying artifacts, flipping a pointer file and recording DB entry.
    Returns the published directory path.
    """
    version_dir = Path(version_dir)
    if not version_dir.exists():
        raise FileNotFoundError(f"Version dir does not exist: {version_dir}")
    logger.info(f"Publishing FAISS version from {version_dir} into root {index_root}")

    # Use a file-level lock so concurrent publishers don't race on pointer writes
    index_root = Path(index_root)
    index_root.mkdir(parents=True, exist_ok=True)
    with acquire_publish_lock(index_root):
        # Verify artifacts
        if verify:
            metadata = _verify_index_dir(version_dir)
        else:
            with open(version_dir / 'metadata.json', 'r', encoding='utf-8') as fh:
                metadata = json.load(fh)

        # Form the published path: keep version_dir as-is; pointer file references it
        published_dir = version_dir

        # Write pointer tmp + os.replace
        pointer_tmp = index_root / (POINTER_FILENAME + '.tmp')
        pointer_final = index_root / POINTER_FILENAME

    pointer_payload = {
        'index_dir': str(published_dir),
        'timestamp': int(time.time()),
    }
    if version_id:
        pointer_payload['version_id'] = version_id
    else:
        pointer_payload['version_id'] = f"faiss_{int(time.time())}"

    # Write pointer to tmp and fsync
    with open(pointer_tmp, 'w', encoding='utf-8') as fh:
        json.dump(pointer_payload, fh)
        fh.flush()
        os.fsync(fh.fileno())

    # Atomically replace pointer; on Windows os.replace is atomic for files. Add a small retry loop to
    # handle transient PermissionError due to AV or file locks.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            os.replace(str(pointer_tmp), str(pointer_final))
            break
        except PermissionError as exc:
            logger.warning(f"Failed to replace pointer file on attempt {attempt}: {exc}")
            if attempt == max_attempts:
                # Move pointer tmp to a failed marker to avoid leaking temp file
                failed_path = index_root / (POINTER_FILENAME + f'.failed.{int(time.time())}')
                os.replace(str(pointer_tmp), str(failed_path))
                raise
            time.sleep(0.1 * attempt)
    logger.info(f"Published pointer file to {pointer_final}")

    # Record in DB
    try:
        manager = get_metadata_manager()
        manager.record_index_version(pointer_payload['version_id'], str(published_dir))
    except Exception as exc:
        logger.warning(f"Failed to record published index version in metadata DB: {exc}")
    return published_dir


def get_current_index_dir(index_root: Path) -> Optional[Path]:
    pointer = Path(index_root) / POINTER_FILENAME
    if not pointer.exists():
        return None
    with open(pointer, 'r', encoding='utf-8') as fh:
        payload = json.load(fh)
    return Path(payload.get('index_dir')) if payload.get('index_dir') else None


def cleanup(index_root: Path, keep_last_n: int = 3):
    """Remove older faiss_v* directories beyond the retention threshold.
    Retention will keep the latest N directories by timestamp embedded in folder name or mtime.
    """
    index_root = Path(index_root)
    if not index_root.exists():
        return
    candidates = [p for p in index_root.iterdir() if p.is_dir() and p.name.startswith('faiss_v')]
    # Sort by mtime to preserve latest
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    to_remove = candidates[keep_last_n:]
    for dr in to_remove:
        try:
            # Safety: only remove directories with prefix faiss_v*
            import shutil
            shutil.rmtree(dr)
            logger.info(f"Removed old FAISS version dir: {dr}")
        except Exception as exc:
            logger.warning(f"Failed to remove old FAISS version dir {dr}: {exc}")
