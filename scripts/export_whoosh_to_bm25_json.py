"""
Export a Whoosh index back to JSON BM25 stats and chunks JSONL (best-effort).
"""

import argparse
import json

from src.cubo.retrieval.bm25_whoosh_store import BM25WhooshStore


def main():
    parser = argparse.ArgumentParser(
        description="Export a Whoosh index to BM25 JSON + chunks JSONL"
    )
    parser.add_argument("--whoosh-dir", required=True, help="Input whoosh index dir")
    parser.add_argument("--out-chunks", required=True, help="Path to output chunks.jsonl")
    args = parser.parse_args()

    store = BM25WhooshStore(index_dir=args.whoosh_dir)

    # Open index and extract docs
    from whoosh import index as whoosh_index

    ix = whoosh_index.open_dir(args.whoosh_dir)
    with ix.searcher() as searcher:
        with open(args.out_chunks, "w", encoding="utf-8") as f:
            for docnum in range(searcher.reader().doc_count_all()):
                try:
                    doc = searcher.stored_fields(docnum)
                    rec = {"doc_id": doc.get("doc_id"), "text": doc.get("text")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    # Skip deleted/invalid docs
                    continue
    print("Exported chunks to", args.out_chunks)


if __name__ == "__main__":
    main()
