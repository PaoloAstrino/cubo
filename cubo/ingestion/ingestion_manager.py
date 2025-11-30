from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from cubo.config import config
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.ingestion.fast_pass_ingestor import build_bm25_index
from cubo.storage.metadata_manager import get_metadata_manager
from cubo.utils.logger import logger


class IngestionManager:
    """Orchestrate fast and deep ingestion runs with status tracking and background processing."""

    def __init__(self):
        self.metadata = get_metadata_manager()
        self.executor = ThreadPoolExecutor(max_workers=config.get("ingestion.deep.n_workers", 2))

    def start_fast_pass(
        self,
        folder: str,
        output_dir: Optional[str] = None,
        skip_model: Optional[bool] = None,
        auto_deep: bool = False,
    ) -> Dict[str, Any]:
        """Run fast pass and optionally trigger background deep ingestion.

        Returns: dict with run_id and fast pass results
        """
        now = int(time.time())
        run_id = f"ingest_{Path(folder).name}_{now}"
        # Build fast pass
        try:
            self.metadata.record_ingestion_run(run_id, folder, 0, None)
        except Exception:
            logger.warning("Failed to record ingestion run (initial)")

        # Run fast pass synchronously
        try:
            result = build_bm25_index(
                folder,
                output_dir,
                skip_model=skip_model or config.get("ingestion.fast_pass.skip_heavy_models", True),
            )
            # Update run details
            try:
                chunks_count = result.get("chunks_count", 0)
                output_path = result.get("chunks_jsonl") or result.get("bm25_stats")
                self.metadata.record_ingestion_run(run_id, folder, chunks_count, output_path)
                self.metadata.update_ingestion_status(
                    run_id,
                    "fast_complete",
                    started_at=datetime.utcnow().isoformat(),
                    finished_at=datetime.utcnow().isoformat(),
                )
            except Exception:
                logger.warning("Failed to update ingestion run details after fast pass")
        except Exception as exc:
            logger.error(f"Fast pass ingestion failed: {exc}")
            self.metadata.update_ingestion_status(run_id, "failed")
            return {"run_id": run_id, "result": None}

        if auto_deep or config.get("ingestion.fast_pass.auto_trigger_deep", False):
            # Trigger deep ingestion in background
            self.executor.submit(self.start_deep_pass, folder, output_dir, run_id)

        return {"run_id": run_id, "result": result}

    def start_deep_pass(
        self,
        folder: str,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Run deep ingestion synchronously (can be used as the background target)."""
        if run_id is None:
            now = int(time.time())
            run_id = f"deep_{Path(folder).name}_{now}"
            try:
                self.metadata.record_ingestion_run(run_id, folder, 0, None)
            except Exception:
                logger.warning("Failed to record deep ingestion run (initial)")

        self.metadata.update_ingestion_status(
            run_id, "deep_started", started_at=datetime.utcnow().isoformat()
        )
        try:
            dee = DeepIngestor(input_folder=folder, output_dir=output_dir)
            result = dee.ingest()
            if result:
                self.metadata.record_ingestion_run(
                    run_id, folder, result.get("chunks_count", 0), result.get("chunks_parquet")
                )
                self.metadata.update_ingestion_status(
                    run_id, "deep_complete", finished_at=datetime.utcnow().isoformat()
                )
            else:
                self.metadata.update_ingestion_status(
                    run_id, "deep_empty", finished_at=datetime.utcnow().isoformat()
                )
            return {"run_id": run_id, "result": result}
        except Exception as exc:
            logger.error(f"Deep ingress failed: {exc}")
            self.metadata.update_ingestion_status(
                run_id, "failed", finished_at=datetime.utcnow().isoformat()
            )
            return {"run_id": run_id, "result": None}

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.metadata.get_ingestion_run(run_id)
        except Exception as exc:
            logger.warning(f"Failed to get run status for {run_id}: {exc}")
            return None


# Path imported above
