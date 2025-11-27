"""
CLI utility to convert JSON BM25 stats + chunks.jsonl to a Whoosh index.
"""

import argparse
import json

from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore


def main():
    parser = argparse.ArgumentParser(
        description="Convert BM25 JSON stats + chunks to a Whoosh index"
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--whoosh-dir", required=True, help="Output whoosh index dir")
    args = parser.parse_args()
    # Load chunks
    docs = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            file_hash = rec.get("file_hash", "")
            chunk_index = rec.get("chunk_index", 0)
            filename = rec.get("filename", "unknown")
            doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
            text = rec.get("text", rec.get("document", ""))
            docs.append({"doc_id": doc_id, "text": text, "metadata": rec})

    store = BM25WhooshStore(index_dir=args.whoosh_dir)
    store.index_documents(docs)
    print("Converted and indexed", len(docs), "docs")


if __name__ == "__main__":
    main()
