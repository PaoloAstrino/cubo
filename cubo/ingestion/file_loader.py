"""
File loader abstraction that supports text/pdf/docx/csv loading in a uniform way.
"""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from cubo.ingestion.document_loader import DocumentLoader
from cubo.utils.logger import logger


class FileLoader:
    """Abstraction that loads supported files and returns "chunks" similar to DocumentLoader.
    Supports CSV by reading a text column or concatenating columns.
    """

    def __init__(self, skip_model: bool = False, text_column: str = None):
        self.doc_loader = DocumentLoader(skip_model=skip_model)
        self.text_column = text_column

    def load_documents_from_folder(self, folder_path: str) -> List[Dict]:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(folder_path)

        # Collect supported files by DocumentLoader
        # DocumentLoader already searches for supported extensions; we'll reuse it for those.
        chunks = self.doc_loader.load_documents_from_folder(folder_path)

        # Additionally process CSV files
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    logger.info(f"Loading CSV file {file_path}")
                    chunks.extend(self._load_csv(file_path))

        return chunks

    def _detect_text_column(self, df, text_col):
        """Detect text column in DataFrame using heuristics."""
        if text_col:
            return text_col

        if "text" in df.columns:
            return "text"
        elif "content" in df.columns:
            return "content"
        else:
            # Fallback to first column with object dtype
            for c in df.columns:
                if df[c].dtype == object:
                    return c
        return None

    def _create_chunk_from_row(self, row, text_col, csv_path, index):
        """Create chunk dictionary from DataFrame row."""
        text = str(row.get(text_col, ""))
        if not text or text.strip() == "":
            return None

        return {
            "filename": Path(csv_path).name,
            "file_hash": "",
            "chunk_index": index,
            "text": text,
            "token_count": len(text.split()),
            "char_length": len(text),
        }

    def _load_csv(self, csv_path: str) -> List[Dict]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Failed to read CSV {csv_path}: {e}")
            return []

        text_col = self._detect_text_column(df, self.text_column)
        if not text_col:
            logger.warning(f"CSV {csv_path} has no obvious text column; skipping")
            return []

        chunks = []
        for i, row in df.iterrows():
            chunk = self._create_chunk_from_row(row, text_col, csv_path, i)
            if chunk:
                chunks.append(chunk)
        return chunks
