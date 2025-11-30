#!/usr/bin/env python3
"""BM25 Query CLI for fast pass chunks and BM25 stats."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.retrieval.bm25_searcher import BM25Searcher


def main():
    parser = argparse.ArgumentParser(description="Query BM25 over fast-pass chunks")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--chunks", default="data/fastpass/chunks.jsonl", help="Chunks JSONL path")
    parser.add_argument("--bm25", default="data/fastpass/bm25_stats.json", help="BM25 stats path")
    parser.add_argument("--topk", type=int, default=10, help="Top k results to return")
    args = parser.parse_args()

    searcher = BM25Searcher(args.chunks, args.bm25)
    results = searcher.search(args.query, args.topk)
    for i, r in enumerate(results, 1):
        filename = r["metadata"].get("filename", "unknown")
        print(f"{i}. {filename} (score: {r['similarity']:.3f})")
        print(f"   {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
