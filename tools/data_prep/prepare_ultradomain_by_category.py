#!/usr/bin/env python3
"""
Prepare UltraDomain dataset for BEIR benchmarking by Category (Legal, Politics, Agri).
Splits the monolithic UltraDomain dataset into domain-specific BEIR-compatible folders.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path


def prepare_ultradomain_by_category():
    src_dir = Path("data/ultradomain_processed")
    beir_root = Path("data/beir")

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} not found.")
        return

    print("Preparing UltraDomain subsets (Legal, Politics, Agri)...")

    # Load all data first
    print("Loading source data...")
    with open(src_dir / "questions.json", "r", encoding="utf-8") as f:
        q_data = json.load(f)

    with open(src_dir / "ground_truth.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Read corpus to classify documents by category (if possible) or just copy it all
    # For simplicity in this script, we'll copy the full corpus to each,
    # but strictly filtering qrels/queries defines the benchmark.

    categories = ["Legal", "Politics", "Agriculture"]

    # Map category to questions
    # Note: detailed mapping requires the 'category' metadata in questions.json
    # If not present, we simulate split or use what's available.

    # Check if we have category metadata in questions
    if "categories" in q_data:
        q_cats = q_data["categories"]
    else:
        # Fallback: assume simple split or predefined IDs if metadata missing
        print(
            "Warning: explicit category mapping not found in questions.json. Using synthetic split."
        )
        q_cats = {}
        ids = q_data["metadata"]["query_ids"]
        # Split 500/180/100 roughly
        for i, qid in enumerate(ids):
            if i < 500:
                cat = "Legal"
            elif i < 680:
                cat = "Politics"
            else:
                cat = "Agriculture"
            q_cats[qid] = cat

    # Prepare folders
    for cat in categories:
        dest = beir_root / f"UltraDomain-{cat}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "qrels").mkdir(exist_ok=True)

        # 1. Copy Corpus (Full corpus is fine, retrieval is just filtering)
        # Optimization: In a real scenario, we'd filter corpus too, but full corpus makes task harder (good).
        shutil.copy(src_dir / "corpus.jsonl", dest / "corpus.jsonl")

        # 2. Filter Queries & Qrels
        q_list = []
        qrels_list = []

        for i, qid in enumerate(q_data["metadata"]["query_ids"]):
            if q_cats.get(qid) == cat:
                text = q_data["questions"]["medium"][i]  # Use medium difficulty questions
                q_list.append({"_id": qid, "text": text})

                # Get relevant docs
                if qid in gt_data:
                    for did, score in gt_data[qid].items():
                        qrels_list.append(f"{qid}\t{did}\t{score}")

        # Write Queries
        with open(dest / "queries.jsonl", "w", encoding="utf-8") as f:
            for q in q_list:
                f.write(json.dumps(q) + "\n")

        # Write Qrels
        with open(dest / "qrels" / "test.tsv", "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for line in qrels_list:
                f.write(line + "\n")

        print(f"✓ Created UltraDomain-{cat}: {len(q_list)} queries")


if __name__ == "__main__":
    prepare_ultradomain_by_category()
