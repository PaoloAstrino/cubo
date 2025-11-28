#!/usr/bin/env python3
"""
Populate FAISS documents.db (sqlite) with documents from a BEIR corpus.jsonl
using existing FAISS index ids so the retriever can return document content
without re-indexing embeddings.

Usage:
python scripts/populate_documents_db_from_beir.py --index-dir ./faiss_index --corpus data/beir/corpus.jsonl
"""
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Populate documents.db from BEIR corpus")
    parser.add_argument("--index-dir", default="./faiss_index", help="Path to FAISS index dir")
    parser.add_argument("--corpus", required=True, help="Path to BEIR corpus.jsonl")
    parser.add_argument("--commit-size", type=int, default=1000, help="Rows to insert per transaction")
    return parser.parse_args()


def load_corpus(corpus_path: Path):
    mapping: Dict[str, str] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            doc_id = str(obj.get("_id") or obj.get("id") or obj.get("doc_id"))
            text = obj.get("text") or obj.get("content") or obj.get("excerpt") or ""
            mapping[doc_id] = text
    return mapping


def open_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def populate_documents_db_from_beir(index_dir: str, corpus_path: str, commit_size: int = 1000) -> int:
    """Populate the documents DB from a BEIR corpus. Returns number of docs inserted."""
    index_dir = Path(index_dir)
    if not index_dir.exists():
        print(f"Index dir not found: {index_dir}")
        return
    db_path = index_dir / "documents.db"
    if not db_path.exists():
        print(f"Creating new documents DB at {db_path}")
    corpus = Path(corpus_path)
    if not corpus.exists():
        print(f"Corpus not found: {corpus}")
        return

    print(f"Loading corpus from {corpus}... (this may take a while)")
    mapping = load_corpus(corpus)
    print(f"Loaded {len(mapping)} documents from corpus")

    conn = open_db(db_path)
    cur = conn.cursor()

    # Determine which ids are present in FAISS index
    # We'll use metadata.json if present to find hot/cold ids
    meta_path = index_dir / "metadata.json"
    ids_to_populate = set()
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as m:
                meta = json.load(m)
                hot = meta.get("hot_ids", []) or []
                cold = meta.get("cold_ids", []) or []
                ids_to_populate = set(hot) | set(cold)
                print(f"Found {len(ids_to_populate)} ids in metadata.json")
        except Exception:
            pass

    # Fallback: we will populate using mapping keys if metadata not usable
    if not ids_to_populate:
        print("metadata.json not usable or empty; using corpus ids as fallback")
        ids_to_populate = set(mapping.keys())

    # Query existing ids in DB to avoid duplicates
    cur.execute("SELECT id FROM documents")
    existing = {row[0] for row in cur.fetchall()}

    # If we have metadata and the DB already contains all documented ids, skip
    if ids_to_populate and len(existing) >= len(ids_to_populate):
        print(f"DB already populated ({len(existing)} rows) >= FAISS id count ({len(ids_to_populate)}). Skipping population.")
        conn.close()
        return 0

    to_insert = []
    for doc_id in ids_to_populate:
        if doc_id in existing:
            continue
        content = mapping.get(doc_id)
        if content is None:
            # If no content was found for doc in corpus, skip
            continue
        metadata = {"filename": corpus.name, "file_path": str(corpus)}
        to_insert.append((doc_id, content, json.dumps(metadata)))

    print(f"Inserting {len(to_insert)} documents into {db_path}")

    # Batch insert
    i = 0
    commit_size = commit_size
    for batch_start in range(0, len(to_insert), commit_size):
        batch = to_insert[batch_start:batch_start + commit_size]
        cur.executemany("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)", batch)
        conn.commit()
        i += len(batch)
        print(f"Committed {i}/{len(to_insert)}")

    # Final counts
    cur.execute("SELECT COUNT(*) FROM documents")
    final_count = cur.fetchone()[0]
    print("Final DB count:", final_count)
    conn.close()
    return final_count

def main():
    args = parse_args()
    populate_documents_db_from_beir(args.index_dir, args.corpus, commit_size=args.commit_size)


if __name__ == '__main__':
    main()
