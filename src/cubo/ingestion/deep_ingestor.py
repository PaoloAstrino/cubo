"""Richer ingestion pipeline for chunk-level processing (moved from src.ingest)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.cubo.config import config
from src.cubo.ingestion.document_loader import DocumentLoader
from src.cubo.utils.logger import logger
from src.cubo.storage.metadata_manager import get_metadata_manager


class DeepIngestor:
    """Full text ingestion flow that produces stable chunk IDs and parquet output."""

    def __init__(
        self,
        input_folder: Optional[str] = None,
        output_dir: Optional[str] = None,
        chunking_config: Optional[Dict[str, Any]] = None,
        csv_rows_per_chunk: Optional[int] = None,
        use_file_hash_for_chunk_id: Optional[bool] = None,
    ):
        self.input_folder = Path(input_folder or config.get("data_folder", "./data"))
        self.output_dir = Path(output_dir or config.get("deep_output_dir", "./data/deep"))
        self.chunking_config = chunking_config or {}
        self.csv_rows_per_chunk = csv_rows_per_chunk or config.get("deep_csv_rows_per_chunk", 25)
        self.use_file_hash_for_chunk_id = (
            use_file_hash_for_chunk_id
            if use_file_hash_for_chunk_id is not None
            else config.get("deep_chunk_id_use_file_hash", True)
        )
        self.loader = DocumentLoader(skip_model=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._supported_extensions = set(self.loader.supported_extensions) | {".csv", ".xlsx"}

    def ingest(self) -> Dict[str, Any]:
        """Process every supported document and persist chunks to parquet."""
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder {self.input_folder} does not exist")

        files = list(self._discover_files())
        processed_files: List[str] = []
        all_chunks: List[Dict[str, Any]] = []

        for path in files:
            chunks = self._process_file(path)
            if chunks:
                all_chunks.extend(chunks)
                processed_files.append(str(path))

        if not all_chunks:
            logger.warning("Deep ingest produced no chunks")
            return {}

        df = pd.DataFrame.from_records(all_chunks)
        parquet_path = self._save_chunks_parquet(df)
        manifest_path = self._write_manifest(len(all_chunks), processed_files)
        # Record ingestion run
        try:
            manager = get_metadata_manager()
            run_id = f"deep_{os.path.basename(str(self.input_folder))}_{int(pd.Timestamp.utcnow().timestamp())}"
            manager.record_ingestion_run(run_id, str(self.input_folder), len(all_chunks), str(parquet_path))
        except Exception:
            logger.warning("Failed to record deep ingestion run to metadata DB")

        return {
            "chunks_parquet": str(parquet_path),
            "manifest": str(manifest_path),
            "chunks_count": len(all_chunks),
            "processed_files": processed_files,
        }

    def _discover_files(self) -> List[Path]:
        """Return every supported file path under the input folder."""
        results: List[Path] = []
        for root, _, names in os.walk(self.input_folder):
            for name in names:
                path = Path(root) / name
                if path.suffix.lower() in self._supported_extensions:
                    results.append(path)
        return results

    def _process_file(self, path: Path) -> List[Dict[str, Any]]:
        ext = path.suffix.lower()
        chunks: List[Dict[str, Any]] = []

        if ext in self.loader.supported_extensions:
            raw_chunks = self.loader.load_single_document(str(path), self.chunking_config)
            chunks = [self._normalize_chunk(chunk, path) for chunk in raw_chunks]
        elif ext == ".csv":
            chunks = [self._normalize_chunk(chunk, path) for chunk in self._process_csv(path)]
        elif ext == ".xlsx":
            chunks = [self._normalize_chunk(chunk, path) for chunk in self._process_excel(path)]
        elif ext == ".pdf":
            chunks = [self._normalize_chunk(chunk, path) for chunk in self._process_pdf(path)]
        else:
            logger.debug(f"Skipping unsupported file {path}")

        return [chunk for chunk in chunks if chunk.get("text")]

    def _process_csv(self, path: Path) -> List[Dict[str, Any]]:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Unable to read CSV {path}: {exc}")
            return []

        if df.empty:
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        for start in range(0, len(df), self.csv_rows_per_chunk):
            block = df.iloc[start : start + self.csv_rows_per_chunk]
            text = block.to_csv(index=False)
            chunks.append(
                {
                    "text": text,
                    "type": "csv",
                    "chunk_index": chunk_index,
                    "row_start": start,
                    "row_end": start + len(block) - 1,
                    "token_count": len(text.split()),
                    "chunk_type": "csv",
                }
            )
            chunk_index += 1

        return chunks

    def _process_excel(self, path: Path) -> List[Dict[str, Any]]:
        try:
            excel = pd.ExcelFile(path)
        except Exception as exc:
            logger.warning(f"Unable to open Excel file {path}: {exc}")
            return []

        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        for sheet_name in excel.sheet_names:
            try:
                df = excel.parse(sheet_name)
            except ValueError:
                continue

            if df.empty:
                continue

            text = f"Sheet: {sheet_name}\n" + df.to_csv(index=False)
            chunks.append(
                {
                    "text": text,
                    "type": "table",
                    "chunk_index": chunk_index,
                    "sheet_name": sheet_name,
                    "table_metadata": {
                        "columns": df.columns.tolist(),
                        "n_rows": len(df),
                        "n_cols": len(df.columns),
                    },
                    "token_count": len(text.split()),
                    "chunk_type": "table",
                }
            )
            chunk_index += 1

        return chunks

    def _process_pdf(self, path: Path) -> List[Dict[str, Any]]:
        """Use pdfplumber for page-wise text + table extraction, fallback to PyPDF2 text extract."""
        try:
            import pdfplumber
        except Exception:
            # Fallback: use loader's default PDF parsing which returns a single text blob
            logger.warning("pdfplumber not available, using fallback PDF extraction")
            raw = self.loader.load_single_document(str(path), self.chunking_config)
            return [
                {
                    "text": c.get("text") or c.get("document"),
                    "type": c.get("type", "text"),
                    "chunk_index": c.get("chunk_index", idx),
                    "sentence_index": c.get("sentence_index"),
                }
                for idx, c in enumerate(raw)
            ]

        chunks: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract page text
                    text = page.extract_text()
                    if text:
                        # Use sentence window chunking across page content if it's large
                        # We'll call Utils.create_sentence_window_chunks for more granular matching
                        from src.cubo.utils.utils import Utils
                        s_chunks = Utils.create_sentence_window_chunks(text, window_size=self.chunking_config.get("window_size", 3))
                        for c in s_chunks:
                            c["type"] = "text"
                            c["page"] = page_num
                            c["page_index"] = page_num
                            c["chunk_index"] = c.get("chunk_index", 0)
                            chunks.append(c)

                    # Extract tables (prettify and add metadata)
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if table:
                            # Build structured CSV text for the table
                            import csv
                            from io import StringIO
                            sio = StringIO()
                            writer = csv.writer(sio)
                            for row in table:
                                # Normalize None entries to empty strings
                                writer.writerow([cell if cell is not None else "" for cell in row])
                            table_text = sio.getvalue()
                            # Basic metadata: rows and columns
                            n_rows = len(table)
                            n_cols = max((len(r) for r in table), default=0)
                            sample_rows = table[:3]
                            chunks.append({
                                "text": table_text,
                                "type": "table",
                                "chunk_index": len(chunks),
                                "page": page_num,
                                "table_index": t_idx,
                                "table_metadata": {
                                    "n_rows": n_rows,
                                    "n_cols": n_cols,
                                    "sample_rows": sample_rows,
                                },
                            })
        except Exception as exc:
            logger.warning(f"Error processing PDF with pdfplumber {path}: {exc}")

        return chunks

    def _normalize_chunk(self, chunk: Dict[str, Any], path: Path) -> Dict[str, Any]:
        record = chunk.copy()
        record.setdefault("chunk_type", record.get("type", "text"))
        record.setdefault("chunk_index", 0)
        record["filename"] = path.name
        record["file_path"] = str(path)
        record.setdefault("token_count", len(record.get("text", "").split()))
        record["file_hash"] = record.get("file_hash") or self.loader._compute_file_hash(str(path))
        # Allow page/table formats for chunk IDs
        if record.get("type") == "table" and record.get("page") is not None and record.get("table_index") is not None:
            record["chunk_id"] = f"{record['file_hash']}_p{record['page']}_t{record['table_index']}"
        elif record.get("type") == "text" and record.get("page") is not None:
            # If we have sentence windows inside pages, we use sentence_index if available
            if record.get("sentence_index") is not None:
                record["chunk_id"] = f"{record['file_hash']}_p{record['page']}_s{record['sentence_index']}"
            else:
                record["chunk_id"] = f"{record['file_hash']}_p{record['page']}_chunk_{record.get('chunk_index', 0)}"
        else:
            record["chunk_id"] = self._make_chunk_id(record)
        return record

    def _make_chunk_id(self, chunk: Dict[str, Any]) -> str:
        base = (
            chunk["file_hash"]
            if self.use_file_hash_for_chunk_id and chunk.get("file_hash")
            else chunk.get("filename", "unknown")
        )

        if chunk.get("chunk_type") == "csv":
            start = chunk.get("row_start", chunk.get("chunk_index", 0))
            return f"{base}_csv_{start}"

        if chunk.get("chunk_type") == "table" and "sheet_name" in chunk:
            sheet = chunk["sheet_name"].replace(" ", "_")
            return f"{base}_sheet_{sheet}"

        sentence_idx = chunk.get("sentence_index")
        if sentence_idx is not None:
            return f"{base}_s{sentence_idx}"

        return f"{base}_chunk_{chunk.get('chunk_index', 0)}"

    def _save_chunks_parquet(self, df: pd.DataFrame) -> Path:
        target = self.output_dir / "chunks_deep.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        try:
            df.to_parquet(tmp, index=False, engine="pyarrow")
        except ImportError as exc:
            logger.error("`pyarrow` is required to write parquet files: %s", exc)
            raise

        os.replace(str(tmp), str(target))
        return target

    def _write_manifest(self, chunks_count: int, files: List[str]) -> Path:
        manifest = {
            "ingested_at": pd.Timestamp.utcnow().isoformat(),
            "source_folder": str(self.input_folder),
            "chunks_count": chunks_count,
            "processed_files": files,
            "created_by": "DeepIngestor",
        }
        manifest_path = self.output_dir / "ingestion_manifest.json"
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        os.replace(str(tmp_manifest), str(manifest_path))
        return manifest_path
def build_deep_index(folder_path: str, output_dir: str = None, skip_model: bool = False, csv_text_column: str = None) -> Dict[str, str]:
    """Compatibility wrapper for legacy callers/tests.

    The newer DeepIngestor implementation lives above (first class in this file)
    and returns a richer dict from `ingest()`. This compatibility wrapper keeps
    the historical `build_deep_index` contract used by integration tests, i.e.
    returning a dict that contains `parquet` and `chunks_count` keys.
    """
    ingestor = DeepIngestor(input_folder=folder_path, output_dir=output_dir, chunking_config=None, csv_rows_per_chunk=None)
    result = ingestor.ingest()
    # Map the modern keys to the older expected return value
    if not result:
        return {}
    return {"parquet": result.get("chunks_parquet"), "chunks_count": result.get("chunks_count")}
