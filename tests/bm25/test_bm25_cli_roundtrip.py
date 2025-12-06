import json
from pathlib import Path

import pytest
pytest.importorskip("torch")

from cubo.retrieval.bm25_python_store import BM25PythonStore


def test_bm25_cli_roundtrip(tmp_path: Path):
    # Create sample chunks JSONL
    chunks_path = tmp_path / "chunks.jsonl"
    docs = [
        {"filename": "a", "file_hash": "", "chunk_index": 0, "text": "apples bananas"},
        {"filename": "b", "file_hash": "", "chunk_index": 0, "text": "cars vehicles"},
        {"filename": "c", "file_hash": "", "chunk_index": 0, "text": "apples car"},
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        for r in docs:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "bm25"
    out_dir.mkdir()
    from cubo.retrieval.bm25_migration import (
        convert_json_stats_to_bm25,
        export_bm25_to_json,
    )

    convert_json_stats_to_bm25(str(tmp_path / "bm25_stats.json"), str(chunks_path), str(out_dir))
    out_chunks = tmp_path / "roundtrip_chunks.jsonl"
    export_bm25_to_json(str(out_dir), str(out_chunks))

    # Compare parity of top-k with Python store
    py_store = BM25PythonStore()
    # Build docs for python store with doc_id and text
    prdocs = [{"doc_id": (r["filename"] + f"_{r['chunk_index']}"), "text": r["text"]} for r in docs]
    py_store.index_documents(prdocs)

    q = "apples"
    py_res = py_store.search(q, top_k=3)
    # Validate by reading exported chunks and indexing in Python store again
    py_store2 = BM25PythonStore()
    with open(out_chunks, encoding="utf-8") as f:
        exported = [json.loads(l) for l in f]
    py_store2.index_documents([{"doc_id": d["doc_id"], "text": d["text"]} for d in exported])
    out_res = py_store2.search(q, top_k=3)
    assert py_res and out_res
    assert py_res[0]["doc_id"] == out_res[0]["doc_id"]
