"""
RAGBench Dataset Preparation Script

This script prepares RAGBench dataset files for use with BEIR evaluation framework.
It converts RAGBench parquet files into the standard BEIR format:

- corpus.jsonl: Documents with IDs and text
- queries.jsonl: Query IDs and text
- qrels/test.tsv: Query-document relevance judgments

The script processes all test split parquet files in the data/ragbench directory
and merges them into a single BEIR-compatible dataset.
"""

import pandas as pd
import json
import hashlib
import os
from pathlib import Path


def prepare_ragbench(limit_per_file=None):
    """Prepare RAGBench dataset for BEIR evaluation.

    Converts RAGBench parquet files to BEIR format by:
    1. Reading all test split parquet files
    2. Creating document corpus with hash-based IDs
    3. Extracting queries and relevance judgments
    4. Saving in BEIR format (corpus.jsonl, queries.jsonl, qrels/test.tsv)

    Args:
        limit_per_file: Optional limit on documents per parquet file for testing
    """
    base_path = Path("data/ragbench")
    dest_dir = Path("data/beir/ragbench_merged")
    dest_dir.mkdir(parents=True, exist_ok=True)

    corpus = {}  # doc_hash -> doc_content
    queries = [] # list of (qid, text, relevant_docs)

    print("Processing RAGBench parquet files...")

    print(f"Searching in {base_path.absolute()}...")
    for fpath in base_path.rglob("*.parquet"):
        print(f"Found candidate: {fpath}")
        # Skip if it's not a test split
        if "test" not in fpath.name.lower():
            continue

        print(f"Reading {fpath.relative_to(base_path)}...")
        df = pd.read_parquet(fpath)

        if limit_per_file:
            df = df.head(limit_per_file)

        for idx, row in df.iterrows():
            question = row["question"]
            docs_list = row["documents"] # list of strings

            qid = str(row["id"]) if "id" in row else f"{fpath.stem}_{idx}"

            relevant_doc_ids = []
            for doc_text in docs_list:
                # Use hash of text as doc_id to handle duplicates
                doc_id = hashlib.md5(doc_text.encode("utf-8")).hexdigest()
                if doc_id not in corpus:
                    corpus[doc_id] = doc_text
                relevant_doc_ids.append(doc_id)

            queries.append({
                "_id": qid,
                "text": question,
                "relevant_ids": relevant_doc_ids
            })

    print(f"Total Queries: {len(queries)}")
    print(f"Total Unique Documents: {len(corpus)}")

    # Save corpus.jsonl - BEIR format for documents
    print("Saving corpus.jsonl...")
    with open(dest_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"_id": doc_id, "text": text, "title": ""}) + "\n")

    # Save queries.jsonl - BEIR format for queries
    print("Saving queries.jsonl...")
    with open(dest_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"_id": q["_id"], "text": q["text"]}) + "\n")

    # Save qrels/test.tsv - BEIR format for relevance judgments
    print("Saving qrels/test.tsv...")
    qrels_dir = dest_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in queries:
            for did in q["relevant_ids"]:
                f.write(f"{q['_id']}\t{did}\t1\n")

    print(f"✓ RAGBench merged ready at {dest_dir}")


if __name__ == "__main__":
    # We might want to limit for the first run if it's too big,
    # but the user said "bench on RAGBench" so let's go full if possible.
    # Total queries seen before was ~7k, which is fine.
    prepare_ragbench()
