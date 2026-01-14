#!/usr/bin/env python3
"""
Prepare category-specific UltraDomain subsets for BEIR benchmarking.
Converts data/ultradomain/{category}.jsonl files into data/beir/ultradomain_{category} format.
"""

import json
import os
import hashlib
from pathlib import Path


def _generate_doc_id(context):
    """Generate document ID from context using MD5 hash."""
    return hashlib.md5(context.encode("utf-8")).hexdigest()


def _generate_query_id(item, question):
    """Generate query ID from item or question hash."""
    return item.get("_id") or hashlib.md5(question.encode("utf-8")).hexdigest()


def _create_corpus_entry(doc_id, item, context):
    """Create a corpus entry dictionary."""
    return {
        "_id": doc_id,
        "title": item.get("meta", {}).get("title", ""),
        "text": context,
        "metadata": item.get("meta", {})
    }


def _create_query_entry(qid, question):
    """Create a query entry dictionary."""
    return {"_id": qid, "text": question}


def _process_jsonl_item(item, corpus, queries, qrels, limit_docs):
    """Process a single JSONL item and update collections."""
    context = item.get("context")
    question = item.get("input")
    
    if not context or not question:
        return False
        
    doc_id = _generate_doc_id(context)
    
    # Add to corpus if not seen and under limit
    if doc_id not in corpus:
        if len(corpus) >= limit_docs:
            return False
        corpus[doc_id] = _create_corpus_entry(doc_id, item, context)
    
    # Add query
    qid = _generate_query_id(item, question)
    queries[qid] = _create_query_entry(qid, question)
    
    # Add qrel
    qrels.append(f"{qid}\t{doc_id}\t1")
    return True


def _load_data_from_source(src_file, corpus, queries, qrels, limit_docs):
    """Load and process all data from source JSONL file."""
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                _process_jsonl_item(item, corpus, queries, qrels, limit_docs)
            except json.JSONDecodeError:
                continue


def _write_corpus(dest_dir, corpus):
    """Write corpus to corpus.jsonl file."""
    print(f"Writing {len(corpus)} documents to corpus.jsonl...")
    with open(dest_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")


def _write_qrels_and_collect_valid_queries(dest_dir, corpus, qrels):
    """Write qrels file and collect valid query IDs."""
    valid_qids = set()
    qrels_dir = dest_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)
    
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f_qrels:
        f_qrels.write("query-id\tcorpus-id\tscore\n")
        for qrel_line in qrels:
            qid, did, score = qrel_line.split("\t")
            if did in corpus:
                f_qrels.write(qrel_line + "\n")
                valid_qids.add(qid)
    
    return valid_qids


def _write_queries(dest_dir, queries, valid_qids):
    """Write queries file for valid query IDs."""
    with open(dest_dir / "queries.jsonl", "w", encoding="utf-8") as f_queries:
        for qid in valid_qids:
            f_queries.write(json.dumps(queries[qid]) + "\n")


def prepare_category(category, limit_docs=2000):
    """Prepare category-specific UltraDomain subset for BEIR benchmarking."""
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
    
    _load_data_from_source(src_file, corpus, queries, qrels, limit_docs)
    _write_corpus(dest_dir, corpus)
    
    print(f"Writing queries and qrels...")
    valid_qids = _write_qrels_and_collect_valid_queries(dest_dir, corpus, qrels)
    _write_queries(dest_dir, queries, valid_qids)

    print(f"✓ UltraDomain ({category}) ready with {len(corpus)} docs and {len(valid_qids)} queries at {dest_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare UltraDomain category for BEIR")
    parser.add_argument("category", type=str, help="Category name (e.g., legal, politics)")
    parser.add_argument("--limit", type=int, default=2000, help="Max unique documents to include")
    
    args = parser.parse_args()
    prepare_category(args.category, args.limit)
