#!/usr/bin/env python3
"""
Prepare UltraDomain dataset for BEIR benchmarking.
Converts data/ultradomain_processed files into data/beir/ultradomain_medium BEIR format.
"""

import json
import shutil
from pathlib import Path


def prepare_ultradomain(difficulty="medium"):
    src_dir = Path("data/ultradomain_processed")
    dest_dir = Path(f"data/beir/ultradomain_{difficulty}")

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} not found.")
        return

    print(f"Preparing UltraDomain ({difficulty}) for BEIR benchmark...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy corpus.jsonl
    print("Copying corpus.jsonl...")
    shutil.copy(src_dir / "corpus.jsonl", dest_dir / "corpus.jsonl")

    # 2. Create queries.jsonl
    print(f"Creating queries.jsonl for {difficulty} questions...")
    with open(src_dir / "questions.json", "r", encoding="utf-8") as f:
        q_data = json.load(f)

    query_ids = q_data["metadata"]["query_ids"]
    questions = q_data["questions"][difficulty]

    with open(dest_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, text in zip(query_ids, questions):
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")

    # 3. Create qrels/test.tsv
    print("Creating qrels/test.tsv...")
    qrels_dir = dest_dir / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    with open(src_dir / "ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        # Only include qrels for the queries we have in our queries.jsonl
        query_set = set(query_ids)
        for qid, docs in gt_data.items():
            if qid in query_set:
                for did, score in docs.items():
                    f.write(f"{qid}\t{did}\t{score}\n")

    print(f"✓ UltraDomain ({difficulty}) ready at {dest_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare UltraDomain for BEIR")
    parser.add_argument(
        "--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"]
    )
    args = parser.parse_args()
    prepare_ultradomain(args.difficulty)
