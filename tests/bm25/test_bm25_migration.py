import json
from pathlib import Path

import pytest

from src.cubo.retrieval.bm25_migration import convert_json_stats_to_whoosh
from src.cubo.retrieval.bm25_python_store import BM25PythonStore

try:

    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False


@pytest.mark.requires_whoosh
@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason="Whoosh not installed")
def test_json_to_whoosh_parity(tmp_path: Path):
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
    out_dir = str(tmp_path / "whoosh")
    convert_json_stats_to_whoosh(str(tmp_path / "bm25_stats.json"), str(chunks_path), out_dir)

    # Compare parity of top-k with Python store
    py_store = BM25PythonStore()
    # Build docs for python store with doc_id and text
    prdocs = [{"doc_id": (r["filename"] + f"_{r['chunk_index']}"), "text": r["text"]} for r in docs]
    py_store.index_documents(prdocs)

    whoosh_store = BM25WhooshStore(index_dir=out_dir)

    q = "apples"
    py_res = py_store.search(q, top_k=3)
    who_res = whoosh_store.search(q, top_k=3)

    # Ensure at least the top doc id matches
    if py_res and who_res:
        assert py_res[0]["doc_id"] == who_res[0]["doc_id"]
