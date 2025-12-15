#!/usr/bin/env python
"""Reindex BEIR corpus using CUBO ingestion pipeline.

This script uses `CuboCore.add_documents` to ingest BEIR corpus into the CUBO index
so that all chunking, embedders, and index settings are applied consistently.

Usage:
  python scripts/reindex_beir_with_cubo.py --corpus data/beir/corpus.jsonl --index-dir results/beir_index --batch-size 500
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cubo.core import CuboCore


def parse_args():
    parser = argparse.ArgumentParser(description="Reindex BEIR corpus with CUBO ingestion")
    parser.add_argument("--corpus", required=True, help="Path to BEIR corpus.jsonl")
    parser.add_argument("--index-dir", required=True, help="Directory to store FAISS vector index")
    parser.add_argument("--batch-size", type=int, default=512, help="Number of docs per batch to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs for testing")
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    index_dir = Path(args.index_dir)
    
    # CRITICAL: Delete existing index completely to prevent stale FAISS index files
    if index_dir.exists():
        print(f"Removing existing index at {index_dir}")
        import shutil
        shutil.rmtree(index_dir, ignore_errors=True)
    
    index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cubo core
    from cubo.config import config

    config.set("vector_store_path", str(index_dir))
    config.set("document_cache_size", 0)

    cubo = CuboCore()
    cubo.initialize_components()

    docs = []
    count = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            try:
                doc = json.loads(line)
                doc_id = str(doc.get('_id', i))
                text = doc.get('text', '')
                title = doc.get('title', '')
                if not text:
                    continue

                docs.append({
                    'text': text,
                    'file_path': f'beir_{doc_id}.txt',
                    'metadata': {'title': title, 'source': 'beir'}
                })
                count += 1

                if len(docs) >= args.batch_size:
                    print(f"Adding batch of {len(docs)} docs (total {count})")
                    cubo.add_documents(docs)
                    docs = []

            except json.JSONDecodeError:
                continue

    if docs:
        print(f"Adding final batch of {len(docs)} docs (total {count})")
        cubo.add_documents(docs)

    print(f"Done. Ingested {count} documents to {index_dir}")


if __name__ == '__main__':
    main()
