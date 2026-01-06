#!/usr/bin/env python3
"""
Prepare category-specific UltraDomain subsets for BEIR benchmarking.
Converts data/ultradomain/{category}.jsonl files into data/beir/ultradomain_{category} format.
"""

import json
import os
import hashlib
from pathlib import Path

def prepare_category(category, limit_docs=2000):
    src_file = Path(f"data/ultradomain/{category}.jsonl")
    dest_dir = Path(f"data/beir/ultradomain_{category}")
    
    if not src_file.exists():
        print(f"Error: Source file {src_file} not found.")
        return

    print(f"Preparing UltraDomain ({category}) subset for BEIR benchmark...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    corpus = {}
    queries = {}
    qrels = []
    
    doc_count = 0
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                context = item.get("context")
                question = item.get("input")
                
                if not context or not question:
                    continue
                    
                # Use hash of context as doc_id for consistency
                doc_id = hashlib.md5(context.encode("utf-8")).hexdigest()
                
                # If we haven't seen this context, add to corpus (if under limit)
                if doc_id not in corpus:
                    if len(corpus) >= limit_docs:
                        continue
                    corpus[doc_id] = {
                        "_id": doc_id,
                        "title": item.get("meta", {}).get("title", ""),
                        "text": context,
                        "metadata": item.get("meta", {})
                    }
                
                # Add query (each input is a query)
                qid = item.get("_id") or hashlib.md5(question.encode("utf-8")).hexdigest()
                queries[qid] = {"_id": qid, "text": question}
                
                # Add qrel
                qrels.append(f"{qid}\t{doc_id}\t1")
                
            except json.JSONDecodeError:
                continue

    # Write corpus.jsonl
    print(f"Writing {len(corpus)} documents to corpus.jsonl...")
    with open(dest_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")
            
    # Write queries.jsonl
    # Only keep queries for which we have the context in our (limited) corpus
    valid_qids = set()
    print(f"Writing queries and qrels...")
    qrels_dir = dest_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f_qrels:
        f_qrels.write("query-id\tcorpus-id\tscore\n")
        # Optimization: only write qrels if the doc is in our corpus
        for qrel_line in qrels:
            qid, did, score = qrel_line.split("\t")
            if did in corpus:
                f_qrels.write(qrel_line + "\n")
                valid_qids.add(qid)
                
    with open(dest_dir / "queries.jsonl", "w", encoding="utf-8") as f_queries:
        for qid in valid_qids:
            f_queries.write(json.dumps(queries[qid]) + "\n")

    print(f"✓ UltraDomain ({category}) ready with {len(corpus)} docs and {len(valid_qids)} queries at {dest_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare UltraDomain category for BEIR")
    parser.add_argument("category", type=str, help="Category name (e.g., legal, politics)")
    parser.add_argument("--limit", type=int, default=2000, help="Max unique documents to include")
    
    args = parser.parse_args()
    prepare_category(args.category, args.limit)
