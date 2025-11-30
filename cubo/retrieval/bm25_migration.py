"""
Migration utilities for BM25 stores using the Python BM25 implementation.

The original code supported Whoosh-based indexes; Whoosh is now deprecated
and removed. These utilities provide conversion and export helpers that work
with the Python BM25 store implementation by saving the pre-built document
chunks and BM25 statistics to disk in a small directory layout.
"""

import json
import os
from pathlib import Path
from typing import Dict


def _read_chunks_jsonl(chunks_jsonl_path: str):
    docs = []
    with open(chunks_jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            file_hash = rec.get("file_hash", "")
            chunk_index = rec.get("chunk_index", 0)
            filename = rec.get("filename", "unknown")
            doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
            text = rec.get("text", rec.get("document", ""))
            docs.append({"doc_id": doc_id, "text": text, "metadata": rec})
    return docs


def convert_json_stats_to_bm25(json_stats_path: str, chunks_jsonl_path: str, output_dir: str) -> Dict:
    """Convert JSON BM25 chunks into a small Python BM25 "index" on disk.

    This function does the following: read input chunks JSONL, construct
    a BM25PythonStore in-memory index (for parity checks), and persist a
    small representation on disk in ``output_dir``. The output directory will
    contain two files:
      - chunks.jsonl: the chunks that were indexed (doc_id+text)
      - bm25_stats.json: BM25 store statistics (if requested via json_stats_path)

    Args:
        json_stats_path: Path to write BM25 statistics JSON (optional).
        chunks_jsonl_path: Source chunks JSONL file path.
        output_dir: Directory to write the exported representation.

    Returns:
        A dict with metadata about conversion.
    """
    from cubo.retrieval.bm25_python_store import BM25PythonStore

    docs = _read_chunks_jsonl(chunks_jsonl_path)
    store = BM25PythonStore()

    # Reformat docs for Python store (doc_id + text only)
    py_docs = [{"doc_id": d["doc_id"], "text": d["text"], "metadata": d.get("metadata", {})} for d in docs]
    store.index_documents(py_docs)

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Persist chunks.jsonl with the same minimal structure as the source
    chunks_out = out_dir_path / "chunks.jsonl"
    with open(chunks_out, "w", encoding="utf-8") as f:
        for d in py_docs:
            f.write(json.dumps({"doc_id": d["doc_id"], "text": d["text"], "metadata": d.get("metadata", {})}, ensure_ascii=False) + "\n")

    # Optionally persist BM25 stats if requested
    stats_out = None
    if json_stats_path:
        stats_out = Path(json_stats_path)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        store.save_stats(str(stats_out))

    return {"docs_indexed": len(py_docs), "bm25_dir": str(out_dir_path), "stats_file": str(stats_out) if stats_out else None}


def export_bm25_to_json(bm25_dir: str, output_chunks_jsonl: str) -> Dict:
    """Export a small BM25 representation (created by convert_json_stats_to_bm25) to chunks JSONL.

    Args:
        bm25_dir: Directory where the small BM25 representation was written.
        output_chunks_jsonl: Destination chunks JSONL file path.

    Returns:
        Metadata dict about the export.
    """
    in_dir = Path(bm25_dir)
    chunks_file = in_dir / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"chunks.jsonl not found in {bm25_dir}")

    with open(output_chunks_jsonl, "w", encoding="utf-8") as out_f, open(chunks_file, encoding="utf-8") as in_f:
        for line in in_f:
            out_f.write(line)

    return {"chunks_exported": True, "output": output_chunks_jsonl}


# Backwards compatible aliases for previous names
convert_json_stats_to_whoosh = convert_json_stats_to_bm25
export_whoosh_to_json = export_bm25_to_json
