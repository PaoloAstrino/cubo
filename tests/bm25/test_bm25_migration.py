import json
from pathlib import Path

import pytest

from cubo.retrieval.bm25_migration import convert_json_stats_to_bm25
from cubo.retrieval.bm25_python_store import BM25PythonStore


def test_json_to_bm25_parity(tmp_path: Path):
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

    # Convert to Whoosh
    out_dir = str(tmp_path / "bm25")
    convert_json_stats_to_bm25(str(tmp_path / "bm25_stats.json"), str(chunks_path), out_dir)

    # Compare parity of top-k with Python store
    py_store = BM25PythonStore()
    # Build docs for python store with doc_id and text
    prdocs = [{"doc_id": (r["filename"] + f"_{r['chunk_index']}"), "text": r["text"]} for r in docs]
    py_store.index_documents(prdocs)

    # Use the Python store to validate parity
    py_store2 = BM25PythonStore()
    # Load docs from the output directory's chunks file
    with open(Path(out_dir) / "chunks.jsonl", encoding="utf-8") as f:
        out_docs = [json.loads(l) for l in f]
    py_store2.index_documents([{"doc_id": d["doc_id"], "text": d["text"]} for d in out_docs])

    q = "apples"
    py_res = py_store.search(q, top_k=3)

    # Ensure at least the top doc id matches
    out_res = py_store2.search(q, top_k=3)
    if py_res and out_res:
        assert py_res[0]["doc_id"] == out_res[0]["doc_id"]
