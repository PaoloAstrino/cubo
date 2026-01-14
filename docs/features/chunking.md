# Chunking Strategies (CUBO)

> **Category:** Feature | **Status:** Active

Overview of chunking strategies available in CUBO, including structure-preserving and sentence-window approaches.

---

This repository now exposes a small set of chunking strategies through a unified interface in `cubo/ingestion/chunkers.py`.

## Strategies
- **StructureChunker**: wraps `HierarchicalChunker` to preserve document structure (sections/paragraphs). Best for .txt/.docx and general text.
- **SentenceWindowChunker**: wraps `Utils.create_sentence_window_chunks`; produces per-sentence chunks with surrounding window context. Used for PDFs and OCR fallbacks; tunable `window_size`.
- **OverlapChunker**: wraps `Utils.chunk_text`; simple character-based overlapping chunks (fallback/legacy).
- **TableChunker**: chunks CSV/Excel (DataFrame) rows into blocks; adds basic metadata.
- **AutoMergingChunkerWrapper**: optional multi-level chunking for dedup/auto-merge retrieval (requires `sentence_transformers`).

## Factory
Use `ChunkerFactory` to obtain a strategy by name or content type:
```python
from cubo.ingestion.chunkers import ChunkerFactory
factory = ChunkerFactory(default_window_size=3, table_rows_per_chunk=25)
text_chunker = factory.for_text()             # StructureChunker
pdf_chunker = factory.for_pdf()               # SentenceWindowChunker
manual = factory.get("overlap")              # OverlapChunker
```

## Current Wiring
- `DocumentLoader` uses `ChunkerFactory.for_text()` (structure-preserving) for plain text/docx.
- `DeepIngestor` uses `ChunkerFactory.for_pdf()` for PDF text/OCR; tables use the existing table path.
- Dedup/auto-merge uses `AutoMergingChunkerWrapper` (specialized path).

## Adding a new strategy
1. Implement a small wrapper in `cubo/ingestion/chunkers.py` implementing `chunk(text, **kwargs)`.
2. Register it in `ChunkerFactory.get` (choose a key name).
3. Add a test covering a simple input/output sample.
4. Update docs if the strategy is user-facing.

## Notes
- Default behavior is unchanged; factory simply centralizes selection.
- Keep `window_size` configurable via settings/config when routing PDFs.
