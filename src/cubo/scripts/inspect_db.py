"""Inspect the configured vector store contents without touching Chroma directly."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.cubo.config import config
from src.cubo.retrieval.vector_store import create_vector_store


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect stored vector data")
    parser.add_argument('--backend', default=None, help='Vector store backend override (faiss|chroma)')
    parser.add_argument('--collection', default=config.get('collection_name', 'cubo_documents'))
    parser.add_argument('--db-path', default=None, help='Override Chroma DB path')
    parser.add_argument('--index-dir', default=None, help='Override FAISS index directory')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.collection:
        config.set('collection_name', args.collection)
    if args.db_path:
        config.set('chroma_db_path', args.db_path)
    if args.index_dir:
        config.set('vector_store_path', args.index_dir)

    store = create_vector_store(
        backend=args.backend,
        collection_name=args.collection,
        index_dir=args.index_dir,
        db_path=args.db_path
    )

    all_data = store.get()

    print("=" * 80)
    print("VECTOR STORE CONTENT INSPECTION")
    print("=" * 80)

    if not all_data or not all_data.get('documents'):
        print("\nNo data in collection")
    else:
        # Group by filename
        file_chunks = defaultdict(list)
        for doc, metadata, doc_id in zip(all_data['documents'], all_data['metadatas'], all_data['ids']):
        filename = metadata.get('filename', 'Unknown')
        file_chunks[filename].append({
            'id': doc_id,
            'text': doc,
            'metadata': metadata
        })

    # Show first 3 chunks from each file
        # Show first 3 chunks from each file
        for filename in sorted(file_chunks.keys()):
        chunks = file_chunks[filename]
        print(f"\n{'='*80}")
        print(f"File: {filename} ({len(chunks)} chunks)")
        print(f"{'='*80}")

        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} (ID: {chunk['id']}):")
            print(f"Text length: {len(chunk['text'])} chars")
            print(f"First 200 chars: {chunk['text'][:200]}")
            print(f"Last 100 chars: ...{chunk['text'][-100:]}")

            if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")


if __name__ == '__main__':
    main()
