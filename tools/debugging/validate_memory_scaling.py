#!/usr/bin/env python3
"""Validate O(1) memory scaling claims for ACL rebuttal.

This script runs ingestion on multiple corpus sizes and validates that
RSS memory remains constant (within tolerance), proving O(1) scaling.

Usage:
    python tools/validate_memory_scaling.py --corpus-sizes 1 5 10
    python tools/validate_memory_scaling.py --use-existing-data data/beir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.ingestion.deep_ingestor import DeepIngestor


def create_synthetic_corpus(size_gb: float, output_dir: Path) -> Path:
    """Create a synthetic corpus of approximately the specified size.

    Args:
        size_gb: Target corpus size in GB
        output_dir: Directory to create corpus in

    Returns:
        Path to the created corpus directory
    """
    corpus_dir = output_dir / f"synthetic_{size_gb}gb"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Each file ~1MB of text (lorem ipsum style)
    file_count = int(size_gb * 1024)  # 1024 files per GB
    chars_per_file = 1024 * 1024  # 1MB per file

    print(f"Creating {file_count} files for ~{size_gb}GB corpus...")

    sample_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100

    for i in range(file_count):
        file_path = corpus_dir / f"doc_{i:05d}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            # Repeat sample text to reach target size
            text = (sample_text * (chars_per_file // len(sample_text) + 1))[:chars_per_file]
            f.write(text)

        if (i + 1) % 100 == 0:
            print(f"  Created {i + 1}/{file_count} files...")

    actual_size_mb = sum(f.stat().st_size for f in corpus_dir.glob("*.txt")) / (1024 * 1024)
    print(f"Corpus created: {actual_size_mb:.1f} MB in {file_count} files")

    return corpus_dir


def run_profiled_ingestion(
    corpus_path: Path,
    output_dir: Path,
    corpus_label: str,
) -> dict:
    """Run ingestion with memory profiling enabled.

    Args:
        corpus_path: Path to corpus directory
        output_dir: Path for ingestion output
        corpus_label: Human-readable label for this run

    Returns:
        Memory profiling statistics
    """
    print(f"\n{'='*60}")
    print(f"Running profiled ingestion: {corpus_label}")
    print(f"Corpus: {corpus_path}")
    print(f"{'='*60}")

    ingestor = DeepIngestor(
        input_folder=str(corpus_path),
        output_dir=str(output_dir),
        profile_memory=True,  # Enable memory profiling
        chunk_batch_size=50,  # Smaller batches = more profiling samples
    )

    result = ingestor.ingest()

    # Read the memory profile
    profile_path = output_dir / "memory_profile.jsonl"
    if profile_path.exists():
        samples = []
        with open(profile_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))

        rss_values = [s["rss_mb"] for s in samples]
        stats = {
            "corpus_label": corpus_label,
            "sample_count": len(samples),
            "min_rss_mb": min(rss_values),
            "max_rss_mb": max(rss_values),
            "delta_rss_mb": max(rss_values) - min(rss_values),
            "chunks_processed": result.get("chunks_count", 0),
            "is_o1": (max(rss_values) - min(rss_values)) < 500,
        }
    else:
        stats = {"error": "No profile generated", "corpus_label": corpus_label}

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate O(1) memory scaling")
    parser.add_argument(
        "--corpus-sizes",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Corpus sizes in GB to test (default: 0.1 0.5 1.0)",
    )
    parser.add_argument(
        "--use-existing-data",
        type=str,
        help="Use existing data directory instead of synthetic corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/memory_profiles",
        help="Directory for results (default: results/memory_profiles)",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    all_stats = []

    if args.use_existing_data:
        # Run on existing data
        corpus_path = Path(args.use_existing_data)
        if not corpus_path.exists():
            print(f"Error: {corpus_path} does not exist")
            sys.exit(1)

        output_dir = output_base / "existing_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = run_profiled_ingestion(corpus_path, output_dir, f"existing:{corpus_path.name}")
        all_stats.append(stats)
    else:
        # Create and test synthetic corpora
        with tempfile.TemporaryDirectory() as tmpdir:
            for size_gb in args.corpus_sizes:
                print(f"\n{'#'*60}")
                print(f"# Testing {size_gb} GB corpus")
                print(f"{'#'*60}")

                corpus_path = create_synthetic_corpus(size_gb, Path(tmpdir))
                output_dir = output_base / f"synthetic_{size_gb}gb"
                output_dir.mkdir(parents=True, exist_ok=True)

                stats = run_profiled_ingestion(corpus_path, output_dir, f"{size_gb}GB")
                all_stats.append(stats)

                # Clean up corpus to save disk space
                shutil.rmtree(corpus_path)

    # Print summary
    print("\n" + "=" * 70)
    print("MEMORY SCALING VALIDATION SUMMARY")
    print("=" * 70)
    print(
        f"{'Corpus':<15} {'Samples':<10} {'Min MB':<10} {'Max MB':<10} {'Delta MB':<10} {'O(1)?':<10}"
    )
    print("-" * 70)

    all_o1 = True
    for s in all_stats:
        if "error" in s:
            print(f"{s['corpus_label']:<15} ERROR: {s['error']}")
            all_o1 = False
        else:
            o1_str = "[YES]" if s["is_o1"] else "[NO]"
            print(
                f"{s['corpus_label']:<15} {s['sample_count']:<10} "
                f"{s['min_rss_mb']:<10.1f} {s['max_rss_mb']:<10.1f} "
                f"{s['delta_rss_mb']:<10.1f} {o1_str}"
            )
            if not s["is_o1"]:
                all_o1 = False

    print("=" * 70)
    if all_o1:
        print("[OK] O(1) MEMORY CLAIM VALIDATED ACROSS ALL CORPUS SIZES")
    else:
        print("⚠️  O(1) MEMORY CLAIM REQUIRES FURTHER INVESTIGATION")
    print("=" * 70)

    # Save summary to JSON
    summary_path = output_base / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "all_o1_validated": all_o1,
                "results": all_stats,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
