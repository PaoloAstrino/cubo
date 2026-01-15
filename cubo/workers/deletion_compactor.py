"""Background deletion compactor worker.

Provides simple APIs to run pending compaction jobs for FaissStore.
"""

from __future__ import annotations

import time

from cubo.utils.logger import logger


def run_compaction_once(vector_store, timeout: int = 300) -> bool:
    """Run ONE compaction cycle for the provided vector_store.

    Returns True if work was performed, False if no pending jobs were found.
    This function is safe to call from a background thread or scheduler.
    """
    if not vector_store:
        raise ValueError("vector_store is required")

    # If store exposes a helper, use it
    try:
        compaction_fn = getattr(vector_store, "_run_compaction_once", None)
        if compaction_fn is None:
            logger.info("Vector store does not support compaction run")
            return False

        start = time.time()
        compaction_fn()
        duration = time.time() - start
        logger.info("Compaction run finished", extra={"duration": duration})
        return True
    except Exception as e:
        logger.error(f"Compaction run failed: {e}")
        return False


def run_compaction_loop(vector_store, interval_seconds: int = 600):
    """Run compaction loop forever with given interval.

    Intended to be launched in a daemon thread by admins or a service manager.
    """
    if not vector_store:
        raise ValueError("vector_store is required")

    logger.info("Starting deletion compaction loop")
    try:
        while True:
            try:
                work_done = run_compaction_once(vector_store)
                if not work_done:
                    # sleep longer when idle
                    time.sleep(interval_seconds)
                else:
                    # brief sleep to allow coalescing further deletes
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"Compaction loop error: {e}")
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info("Compaction loop interrupted, exiting")
