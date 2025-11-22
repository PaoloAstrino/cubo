"""
File loader abstraction that supports text/pdf/docx/csv loading in a uniform way.
"""
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.cubo.ingestion.document_loader import DocumentLoader
from src.cubo.utils.logger import logger


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
                if filename.lower().endswith('.csv'):
                    file_path = os.path.join(root, filename)
                    logger.info(f"Loading CSV file {file_path}")
                    chunks.extend(self._load_csv(file_path))

        return chunks

    def _load_csv(self, csv_path: str) -> List[Dict]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.warning(f"Failed to read CSV {csv_path}: {e}")
            return []

        text_col = self.text_column
        if not text_col:
            # Heuristic: try `text` or `content` or first string column
            if 'text' in df.columns:
                text_col = 'text'
            elif 'content' in df.columns:
                text_col = 'content'
            else:
                # Fallback to first column with object dtype
                text_col = None
                for c in df.columns:
                    if df[c].dtype == object:
                        text_col = c
                        break
        if not text_col:
            logger.warning(f"CSV {csv_path} has no obvious text column; skipping")
            return []

        chunks = []
        for i, row in df.iterrows():
            text = str(row.get(text_col, ''))
            if not text or text.strip() == '':
                continue
            # Minimal chunk metadata to align with DocumentLoader output
            chunks.append({
                'filename': Path(csv_path).name,
                'file_hash': '',
                'chunk_index': i,
                'text': text,
                'token_count': len(text.split()),
                'char_length': len(text)
            })
        return chunks
