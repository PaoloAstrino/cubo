"""
BEIR Dataset Counter

This script analyzes BEIR (Benchmarking Information Retrieval) datasets in the data/beir directory.
It counts documents, queries, and checks for qrels files, then generates a manifest CSV.

Output:
- results/dataset_manifest.csv: Detailed dataset information
- Console table: Summary of all datasets found
"""

import csv
import os


def count_lines(filepath):
    """Count the number of lines in a file.

    Args:
        filepath: Path to the file to count lines in

    Returns:
        Number of lines, or 0 if file cannot be read
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def analyze_dataset(root, dataset_name):
    """Analyze a single BEIR dataset directory.

    Checks for standard BEIR files (corpus.jsonl, queries.jsonl, qrels/)
    and counts their contents.

    Args:
        root: Root directory containing datasets
        dataset_name: Name of the dataset subdirectory

    Returns:
        Dictionary with dataset analysis results
    """
    # Standard BEIR structure
    corpus_path = os.path.join(root, dataset_name, "corpus.jsonl")
    queries_path = os.path.join(root, dataset_name, "queries.jsonl")
    qrels_path = os.path.join(root, dataset_name, "qrels")

    # Check for direct qrels file if folder doesn't exist
    if not os.path.exists(qrels_path):
        # some datasets might have qrels/test.tsv inside dataset folder
        # or just be missing
        pass

    results = {
        "dataset": dataset_name,
        "path": os.path.join(root, dataset_name),
        "corpus_exists": os.path.exists(corpus_path),
        "queries_exists": os.path.exists(queries_path),
        "qrels_exists": os.path.exists(qrels_path) and os.listdir(qrels_path),
        "corpus_count": count_lines(corpus_path) if os.path.exists(corpus_path) else 0,
        "queries_count": count_lines(queries_path) if os.path.exists(queries_path) else 0,
    }
    return results


def main():
    """Main function to analyze all BEIR datasets."""
    root = "data/beir"
    if not os.path.exists(root):
        print(f"Root {root} not found")
        return

    datasets = [
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ]

    manifest_data = []
    print(f"Found {len(datasets)} potential datasets in {root}")

    for d in datasets:
        print(f"Analyzing {d}...")
        info = analyze_dataset(root, d)
        manifest_data.append(info)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    with open("results/dataset_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_data[0].keys())
        writer.writeheader()
        writer.writerows(manifest_data)

    print("Manifest saved to results/dataset_manifest.csv")

    # Print table
    print(f"{'Dataset':<25} | {'Docs':<10} | {'Queries':<10} | {'Qrels':<5}")
    print("-" * 60)
    for info in manifest_data:
        print(
            f"{info['dataset']:<25} | {info['corpus_count']:<10} | {info['queries_count']:<10} | {'Yes' if info['qrels_exists'] else 'No':<5}"
        )


if __name__ == "__main__":
    main()
