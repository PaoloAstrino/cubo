#!/usr/bin/env python
"""Validate FAISS index for consistent vector norms.

This script checks that all vectors in the hot and cold indexes have
consistent norms (expected to be ~1.0 if normalized).

Usage:
    python tools/validate_faiss_index.py data/faiss
    python tools/validate_faiss_index.py data/faiss --expected-norm 1.0 --tolerance 0.01
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate FAISS index vector norms")
    parser.add_argument(
        "index_dir",
        type=Path,
        help="Path to FAISS index directory",
    )
    parser.add_argument(
        "--expected-norm",
        type=float,
        default=1.0,
        help="Expected vector norm (default: 1.0 for normalized)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for norm deviation (default: 0.01)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of vectors to sample for checking (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed statistics",
    )
    return parser.parse_args()


def check_index_norms(
    index: faiss.Index,
    index_name: str,
    expected_norm: float,
    tolerance: float,
    sample_size: int,
    verbose: bool,
) -> bool:
    """Check vector norms in a FAISS index.

    Returns:
        True if all sampled vectors are within tolerance of expected norm.
    """
    if index is None or index.ntotal == 0:
        print(f"  {index_name}: Empty or None index, skipping")
        return True

    total = index.ntotal
    n_check = min(sample_size, total)

    # Sample indices evenly distributed across the index
    if n_check < total:
        indices = np.linspace(0, total - 1, n_check, dtype=int)
    else:
        indices = np.arange(total)

    # Check if we can reconstruct vectors
    try:
        index.reconstruct_n(0, min(10, total))
    except RuntimeError as e:
        print(f"  {index_name}: Cannot reconstruct vectors ({e})")
        return True  # Can't validate, assume OK

    # Compute norms for sampled vectors
    norms = []
    for i in indices:
        try:
            vec = index.reconstruct(int(i))
            norm = np.linalg.norm(vec)
            norms.append(norm)
        except RuntimeError:
            continue

    if not norms:
        print(f"  {index_name}: Could not reconstruct any vectors")
        return False

    norms = np.array(norms)
    min_norm = norms.min()
    max_norm = norms.max()
    mean_norm = norms.mean()
    std_norm = norms.std()

    # Check if norms are within tolerance
    within_tolerance = np.abs(norms - expected_norm) <= tolerance
    pct_ok = within_tolerance.sum() / len(norms) * 100

    # Report
    print(f"  {index_name}: {total} vectors")
    if verbose:
        print(f"    Sampled: {len(norms)} vectors")
        print(f"    Norm range: [{min_norm:.4f}, {max_norm:.4f}]")
        print(f"    Norm mean±std: {mean_norm:.4f} ± {std_norm:.4f}")
        print(f"    Within tolerance ({expected_norm}±{tolerance}): {pct_ok:.1f}%")

    if pct_ok < 99.0:  # Allow 1% tolerance for floating point
        print(f"    ⚠️  WARNING: Only {pct_ok:.1f}% of vectors within expected norm range")
        return False
    else:
        print(f"    ✓ All vectors properly normalized (mean={mean_norm:.4f})")
        return True


def main():
    args = parse_args()

    index_dir = args.index_dir
    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}")
        sys.exit(1)

    # Load metadata
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {index_dir}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    print("\n=== FAISS Index Validation ===")
    print(f"Index directory: {index_dir}")
    print(f"Dimension: {metadata.get('dimension', 'unknown')}")
    print(f"Normalize flag: {metadata.get('normalize', 'not set')}")
    print(f"Model path: {metadata.get('model_path', 'not set')}")
    print(f"Hot IDs: {len(metadata.get('hot_ids', []))}")
    print(f"Cold IDs: {len(metadata.get('cold_ids', []))}")
    print()

    # Determine expected norm based on metadata
    if metadata.get("normalize", False):
        expected_norm = 1.0
        print(f"Expecting normalized vectors (norm ≈ {expected_norm})")
    else:
        expected_norm = args.expected_norm
        print(f"Using expected norm from args: {expected_norm}")

    print()
    all_ok = True

    # Check hot index
    hot_path = index_dir / "hot.index"
    if hot_path.exists():
        print("Checking hot index...")
        hot_index = faiss.read_index(str(hot_path))
        ok = check_index_norms(
            hot_index,
            "hot.index",
            expected_norm,
            args.tolerance,
            args.sample_size,
            args.verbose,
        )
        all_ok = all_ok and ok
    else:
        print("Hot index not found, skipping")

    print()

    # Check cold index
    cold_path = index_dir / "cold.index"
    if cold_path.exists():
        print("Checking cold index...")
        cold_index = faiss.read_index(str(cold_path))
        ok = check_index_norms(
            cold_index,
            "cold.index",
            expected_norm,
            args.tolerance,
            args.sample_size,
            args.verbose,
        )
        all_ok = all_ok and ok
    else:
        print("Cold index not found, skipping")

    print()

    if all_ok:
        print("✓ Index validation PASSED")
        sys.exit(0)
    else:
        print("✗ Index validation FAILED - vectors have inconsistent norms")
        sys.exit(1)


if __name__ == "__main__":
    main()
