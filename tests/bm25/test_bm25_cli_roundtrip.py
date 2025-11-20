import json
from pathlib import Path
import pytest
from src.cubo.retrieval.bm25_python_store import BM25PythonStore

pytest.importorskip('whoosh')


@pytest.mark.requires_whoosh
def test_bm25_cli_roundtrip(tmp_path: Path):
    # Create sample chunks JSONL
    chunks_path = tmp_path / 'chunks.jsonl'
    docs = [
        {'filename': 'a', 'file_hash': '', 'chunk_index': 0, 'text': 'apples bananas'},
        {'filename': 'b', 'file_hash': '', 'chunk_index': 0, 'text': 'cars vehicles'},
        {'filename': 'c', 'file_hash': '', 'chunk_index': 0, 'text': 'apples car'}
    ]
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for r in docs:
            f.write(json.dumps(r) + '\n')

    out_dir = tmp_path / 'whoosh'
    out_dir.mkdir()
    from src.cubo.retrieval.bm25_migration import convert_json_stats_to_whoosh, export_whoosh_to_json

    convert_json_stats_to_whoosh(str(tmp_path / 'bm25_stats.json'), str(chunks_path), str(out_dir))
    out_chunks = tmp_path / 'roundtrip_chunks.jsonl'
    export_whoosh_to_json(str(out_dir), str(out_chunks))

    # Compare parity of top-k with Python store
    py_store = BM25PythonStore()
    # Build docs for python store with doc_id and text
    prdocs = [{'doc_id': (r['filename'] + f"_{r['chunk_index']}"), 'text': r['text']} for r in docs]
    py_store.index_documents(prdocs)

    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
    whoosh_store = BM25WhooshStore(index_dir=str(out_dir))

    q = 'apples'
    py_res = py_store.search(q, top_k=3)
    who_res = whoosh_store.search(q, top_k=3)

    assert py_res and who_res
    assert py_res[0]['doc_id'] == who_res[0]['doc_id']
