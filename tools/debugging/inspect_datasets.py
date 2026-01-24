"""
Dataset Inspection Utilities

This script provides inspection tools for various datasets used in the project:
- UltraDomain: JSONL format with questions and contexts
- RAGBench: Parquet format with questions and document collections

Provides statistics on file sizes, question counts, and context uniqueness.
"""

import json
import os
from pathlib import Path
import pandas as pd


def inspect_ultradomain():
    """Inspect UltraDomain dataset files and provide statistics.

    Scans all JSONL files in data/ultradomain and reports:
    - File size in MB
    - Number of questions per category
    - Number of unique contexts per category
    - Total counts across all categories
    """
    print("--- UltraDomain Inspection (data/ultradomain) ---")
    total_q = 0
    total_c = 0
    base_path = Path("data/ultradomain")
    if not base_path.exists():
        print("data/ultradomain not found")
        return

    for fpath in base_path.glob("*.jsonl"):
        cat = fpath.stem
        q_count = 0
        contexts = set()
        file_size = os.path.getsize(fpath) / (1024 * 1024)

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q_count += 1
                    contexts.add(item.get("context", ""))
                except:
                    continue

        print(f"{cat:15} | File: {file_size:6.1f}MB | Questions: {q_count:6} | Contexts: {len(contexts):5}")
        total_q += q_count
        total_c += len(contexts)

    print(f"{'TOTAL':15} | {'':11} | Questions: {total_q:6} | Contexts: {total_c:5}")
    print("\n")


def inspect_ragbench():
    """Inspect RAGBench dataset files and provide statistics.

    Scans all parquet files in data/ragbench and reports:
    - File size in MB
    - Number of questions per file
    - Number of unique contexts (or indicates list format)
    """
    print("--- RAGBench Inspection (data/ragbench) ---")
    base_path = Path("data/ragbench")
    if not base_path.exists():
        print("data/ragbench not found")
        return

    for fpath in base_path.glob("*.parquet"):
        try:
            df = pd.read_parquet(fpath)
            # Try to identify question/context columns
            # RAGBench usually has 'question', 'context', or 'documents'
            q_col = next((c for c in df.columns if 'question' in c.lower() or 'query' in c.lower()), None)
            c_col = next((c for c in df.columns if 'context' in c.lower() or 'doc' in c.lower() or 'passage' in c.lower()), None)

            file_size = os.path.getsize(fpath) / (1024 * 1024)
            n_q = len(df)
            n_c = 0
            if c_col:
                # If context is a string
                if isinstance(df[c_col].iloc[0], str):
                    n_c = df[c_col].nunique()
                # If context is a list of docs
                elif isinstance(df[c_col].iloc[0], (list, tuple)):
                    n_c = "N/A (List of docs)"

            print(f"{fpath.name:30} | Size: {file_size:5.1f}MB | Questions: {n_q:6} | Contexts: {n_c}")
        except Exception as e:
            print(f"{fpath.name:30} | Error: {e}")


if __name__ == "__main__":
    inspect_ultradomain()
    try:
        inspect_ragbench()
    except Exception as e:
        print(f"RAGBench Error: {e}")
