"""Unified chunker interfaces and factory.

Provides thin wrappers around existing chunking strategies so callers can
select strategies by name or content type without duplicating logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from cubo.ingestion.hierarchical_chunker import HierarchicalChunker
from cubo.utils.utils import Utils

try:
    # Optional dependency: used only for dedup/auto-merge path
    from cubo.deduplication.custom_auto_merging import AutoMergingChunker as _AutoMergingChunker
except Exception:  # pragma: no cover - optional
    _AutoMergingChunker = None


class BaseChunker(Protocol):
    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        ...


class StructureChunker:
    """Wrapper for the structure-preserving hierarchical chunker."""

    def __init__(self, **kwargs):
        self._chunker = HierarchicalChunker(**kwargs)

    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        return self._chunker.chunk(text, **kwargs)


class SentenceWindowChunker:
    """Sentence-window chunking using Utils.create_sentence_window_chunks."""

    def __init__(self, window_size: int = 3, tokenizer_name: Optional[str] = None):
        self.window_size = window_size
        self.tokenizer_name = tokenizer_name

    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        # kwargs may override window_size/tokenizer_name
        w = kwargs.get("window_size", self.window_size)
        t = kwargs.get("tokenizer_name", self.tokenizer_name)
        return Utils.create_sentence_window_chunks(text, window_size=w, tokenizer_name=t)


class OverlapChunker:
    """Simple overlapping character-based chunker using Utils.chunk_text."""

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        cs = kwargs.get("chunk_size", self.chunk_size)
        ov = kwargs.get("overlap", self.overlap)
        chunks = Utils.chunk_text(text, chunk_size=cs, overlap=ov)
        return [{"text": c, "chunk_index": i, "chunk_type": "text"} for i, c in enumerate(chunks)]


class TableChunker:
    """Chunk tabular data into row blocks (CSV/Excel)."""

    def __init__(self, rows_per_chunk: int = 25):
        self.rows_per_chunk = rows_per_chunk

    def chunk(self, dataframe, chunk_type: str = "table", metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # dataframe expected to be pandas DataFrame
        import pandas as pd  # local import to avoid hard dependency at module import time

        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            return []

        df_str = dataframe.astype(str)
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0
        meta = metadata or {}

        for start in range(0, len(df_str), self.rows_per_chunk):
            block = df_str.iloc[start : start + self.rows_per_chunk]
            text = block.to_csv(index=False)
            if chunk_type == "table" and meta.get("sheet_name"):
                text = f"Sheet: {meta['sheet_name']}\n" + text

            chunk_data = {
                "text": text,
                "type": chunk_type,
                "chunk_index": chunk_index,
                "row_start": start,
                "row_end": start + len(block) - 1,
                "token_count": len(text.split()),
                "chunk_type": chunk_type,
            }
            if meta:
                chunk_data.update(meta)
            chunks.append(chunk_data)
            chunk_index += 1

        return chunks


class AutoMergingChunkerWrapper:
    """Optional wrapper for the auto-merging multi-level chunker (dedup)."""

    def __init__(self, chunk_sizes: Optional[List[int]] = None):
        self._chunker = _AutoMergingChunker(chunk_sizes=chunk_sizes) if _AutoMergingChunker else None

    def chunk(self, text: str, filename: str = "unknown", **kwargs) -> List[Dict[str, Any]]:
        if not self._chunker:
            return []
        return self._chunker.create_hierarchical_chunks(text, filename)


@dataclass
class ChunkerFactory:
    """Factory to obtain chunkers by name or by simple heuristics."""

    default_window_size: int = 3
    table_rows_per_chunk: int = 25

    def get(self, name: str) -> BaseChunker:
        key = (name or "").lower()
        if key in {"structure", "hierarchical", "structured"}:
            return StructureChunker()
        if key in {"sentence_window", "sentence-window", "sentence"}:
            return SentenceWindowChunker(window_size=self.default_window_size)
        if key in {"overlap", "char", "character"}:
            return OverlapChunker()
        if key in {"table", "csv", "excel"}:
            return TableChunker(rows_per_chunk=self.table_rows_per_chunk)
        if key in {"auto_merge", "automerging", "dedup"}:
            return AutoMergingChunkerWrapper()
        # fallback
        return StructureChunker()

    def for_pdf(self) -> BaseChunker:
        return SentenceWindowChunker(window_size=self.default_window_size)

    def for_text(self) -> BaseChunker:
        return StructureChunker()

    def for_table(self) -> BaseChunker:
        return TableChunker(rows_per_chunk=self.table_rows_per_chunk)


# Convenience exports
__all__ = [
    "BaseChunker",
    "StructureChunker",
    "SentenceWindowChunker",
    "OverlapChunker",
    "TableChunker",
    "AutoMergingChunkerWrapper",
    "ChunkerFactory",
]
