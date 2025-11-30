import json
from pathlib import Path

from cubo.retrieval.bm25_searcher import BM25Searcher


def test_searcher_loads_chunks(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    docs = [
        {"filename": "a", "file_hash": "", "chunk_index": 0, "text": "apples"},
        {"filename": "b", "file_hash": "", "chunk_index": 0, "text": "cars"},
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        for r in docs:
            f.write(json.dumps(r) + "\n")
    bs = BM25Searcher(chunks_jsonl=str(chunks_path))
    assert len(bs.docs) == 2
    res = bs.search("apples")
    assert res and res[0]["doc_id"].endswith("_0")
