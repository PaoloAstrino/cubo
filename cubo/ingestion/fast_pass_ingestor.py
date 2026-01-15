"""
FastPass ingestion module
"""

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from cubo.config import config
from cubo.ingestion.document_loader import DocumentLoader
from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.storage.metadata_manager import get_metadata_manager
from cubo.utils.logger import logger


class FastPassIngestor:
    """Quick ingestion path to make documents queryable ASAP with BM25."""

    def __init__(self, output_dir: str = None, skip_model: bool = False):
        self.output_dir = Path(
            output_dir
            or config.get(
                "ingestion.fast_pass.output_dir",
                config.get("fast_pass_output_dir", "data/fastpass"),
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = DocumentLoader(skip_model=skip_model)
        self.skip_model = skip_model

    def _build_records_from_chunks(self, chunks: list) -> tuple:
        """Build records, texts, and doc_ids from chunks."""
        records = []
        texts = []
        doc_ids = []

        for c in chunks:
            filename = c.get("filename", "unknown")
            file_hash = c.get("file_hash", "")
            chunk_index = c.get("chunk_index", 0)
            text = c.get("text", "") or c.get("document", "")
            token_count = c.get("token_count", len(text.split()))

            records.append(
                {
                    "filename": filename,
                    "file_hash": file_hash,
                    "chunk_index": chunk_index,
                    "text": text,
                    "token_count": token_count,
                    "char_length": len(text),
                }
            )
            texts.append(text)
            doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
            doc_ids.append(doc_id)

        return records, texts, doc_ids

    def _save_chunks_jsonl(self, records: list) -> Path:
        """Save chunks to JSONL file atomically."""
        chunks_jsonl = self.output_dir / "chunks.jsonl.tmp"
        final_chunks_jsonl = self.output_dir / "chunks.jsonl"

        with open(chunks_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        os.replace(str(chunks_jsonl), str(final_chunks_jsonl))
        logger.info(f"Saved {len(records)} chunks to {final_chunks_jsonl}")
        return final_chunks_jsonl

    def _build_bm25_index(
        self, texts: list, doc_ids: list, folder_path: str, records: list, final_chunks_jsonl: Path
    ) -> Optional[Path]:
        """Build BM25 index and save stats."""
        try:
            from cubo.config import config

            backend = config.get("bm25.backend", "python")
            bm25 = BM25Searcher(backend=backend)
            docs = [{"doc_id": did, "text": txt} for did, txt in zip(doc_ids, texts)]
            bm25.index_documents(docs)

            bm25_tmp = self.output_dir / "bm25_stats.json.tmp"
            bm25_path = self.output_dir / "bm25_stats.json"
            bm25.save_stats(str(bm25_tmp))
            os.replace(str(bm25_tmp), str(bm25_path))

            # Record ingestion run
            try:
                manager = get_metadata_manager()
                run_id = f"fastpass_{os.path.basename(str(Path(folder_path)))}_{int(pd.Timestamp.utcnow().timestamp())}"
                manager.record_ingestion_run(
                    run_id, str(folder_path), len(records), str(final_chunks_jsonl)
                )
            except Exception:
                logger.warning("Failed to record ingestion run to metadata DB")

            return bm25_path
        except Exception as e:
            logger.error(f"Failed to build BM25 stats: {e}")
            return None

    def _save_manifest(self, folder_path: str, record_count: int):
        """Save ingestion manifest."""
        manifest = {
            "ingested_at": pd.Timestamp.utcnow().isoformat(),
            "source_folder": folder_path,
            "chunks_count": record_count,
            "created_by": "FastPassIngestor",
            "skip_model": self.skip_model,
        }
        manifest_path = self.output_dir / "ingestion_manifest.json.tmp"
        final_manifest_path = self.output_dir / "ingestion_manifest.json"

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
        os.replace(str(manifest_path), str(final_manifest_path))

    def ingest_folder(self, folder_path: str) -> dict:
        """Ingest a folder quickly and create a chunks parquet + BM25 stats.

        Returns: dict with paths {"chunks_parquet": path, "bm25_stats": path}
        """
        logger.info(f"Fast pass ingest start: {folder_path}")

        if self.skip_model:
            self.loader.enhanced_processor = None

        chunks = self.loader.load_documents_from_folder(folder_path)
        if not chunks:
            logger.warning("No chunks produced in fast pass ingest")
            return {}

        # Build records and save
        records, texts, doc_ids = self._build_records_from_chunks(chunks)
        final_chunks_jsonl = self._save_chunks_jsonl(records)

        # Build BM25 index
        bm25_path = self._build_bm25_index(texts, doc_ids, folder_path, records, final_chunks_jsonl)

        # Save manifest
        self._save_manifest(folder_path, len(records))

        return {
            "chunks_jsonl": str(final_chunks_jsonl),
            "bm25_stats": str(bm25_path) if bm25_path else None,
            "chunks_count": len(records),
        }
        """Ingest a folder quickly and create a chunks parquet + BM25 stats.

        Returns: dict with paths {"chunks_parquet": path, "bm25_stats": path}
        """
        logger.info(f"Fast pass ingest start: {folder_path}")

        if self.skip_model:
            self.loader.enhanced_processor = None

        chunks = self.loader.load_documents_from_folder(folder_path)
        if not chunks:
            logger.warning("No chunks produced in fast pass ingest")
            return {}

        # Build records and save
        records, texts, doc_ids = self._build_records_from_chunks(chunks)
        final_chunks_jsonl = self._save_chunks_jsonl(records)

        # Build BM25 index
        bm25_path = self._build_bm25_index(texts, doc_ids, folder_path, records, final_chunks_jsonl)

        # Save manifest
        self._save_manifest(folder_path, len(records))

        return {
            "chunks_jsonl": str(final_chunks_jsonl),
            "bm25_stats": str(bm25_path) if bm25_path else None,
            "chunks_count": len(records),
        }


def build_bm25_index(folder_path: str, output_dir: str = None, skip_model: bool = False) -> dict:
    ingestor = FastPassIngestor(output_dir=output_dir, skip_model=skip_model)
    return ingestor.ingest_folder(folder_path)
