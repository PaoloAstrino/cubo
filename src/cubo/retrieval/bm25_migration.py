"""
Migration utilities for BM25 stores, e.g., converting JSON stats to Whoosh index and vice-versa.
"""
import json
from typing import List
from pathlib import Path

def convert_json_stats_to_whoosh(json_stats_path: str, chunks_jsonl_path: str, output_whoosh_dir: str):
    """Convert JSON BM25 chunks and create a Whoosh index in output_whoosh_dir.

    This function relies on `BM25WhooshStore` being available.
    """
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore

    docs = []
    with open(chunks_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            file_hash = rec.get('file_hash', '')
            chunk_index = rec.get('chunk_index', 0)
            filename = rec.get('filename', 'unknown')
            doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
            text = rec.get('text', rec.get('document', ''))
            docs.append({'doc_id': doc_id, 'text': text, 'metadata': rec})

    store = BM25WhooshStore(index_dir=output_whoosh_dir)
    store.index_documents(docs)
    return {'docs_indexed': len(docs), 'whoosh_dir': output_whoosh_dir}


def export_whoosh_to_json(whoosh_dir: str, output_chunks_jsonl: str):
    """Export a Whoosh index to a chunks JSONL file (best-effort).
    """
    from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore
    from whoosh import index as whoosh_index

    ix = whoosh_index.open_dir(whoosh_dir)
    with ix.searcher() as searcher:
        with open(output_chunks_jsonl, 'w', encoding='utf-8') as f:
            # Iterate through all documents in the Whoosh index using reader
            for docnum in range(searcher.reader().doc_count_all()):
                try:
                    doc = searcher.stored_fields(docnum)
                    rec = {'doc_id': doc.get('doc_id'), 'text': doc.get('text')}
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                except Exception:
                    # Skip deleted or invalid documents
                    continue
    return {'chunks_exported': True, 'output': output_chunks_jsonl}
