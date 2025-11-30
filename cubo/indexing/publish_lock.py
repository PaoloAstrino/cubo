"""
Publisher file lock helper to serialize index publications.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock


@contextmanager
def acquire_publish_lock(index_root: Path, timeout: int = 30) -> Iterator[FileLock]:
    """Acquire a file lock for publishing to index_root and yield a FileLock object.

    Usage:
        with acquire_publish_lock(index_root):
            publish_version(...)
    """
    lockfile = Path(index_root) / "publish.lock"
    lock = FileLock(str(lockfile))
    try:
        lock.acquire(timeout=timeout)
        yield lock
    finally:
        try:
            lock.release()
        except Exception:
            pass
