#!/usr/bin/env python3
"""
Download UltraDomain Subsets

Downloads agriculture, legal, CS, and mix subsets from HuggingFace.
Total size: ~3-4 GB

Usage:
    python scripts/download_ultradomain_subsets.py --output-dir data/ultradomain_hf
"""

import argparse
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Install with: pip install datasets")
    exit(1)


def download_ultradomain_subsets(output_dir: str, subsets: list = None):
    """Download UltraDomain subsets from HuggingFace.
    
    Args:
        output_dir: Directory to save the datasets
        subsets: List of subsets to download (agriculture, legal, cs, mix)
    """
    if subsets is None:
        subsets = ["agriculture", "legal", "cs", "mix"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading UltraDomain subsets from HuggingFace...")
    print(f"Subsets: {', '.join(subsets)}")
    print("This may take 10-20 minutes...\n")
    
    for subset in subsets:
        try:
            print(f"\n{'='*60}")
            print(f"Downloading: {subset}")
            print(f"{'='*60}")
            
            # Load dataset
            dataset = load_dataset("TommyChien/UltraDomain", subset)
            
            # Save to disk
            subset_path = output_path / f"ultradomain_{subset}"
            dataset.save_to_disk(str(subset_path))
            
            # Print info
            print(f"\n[OK] Downloaded {subset}")
            print(f"     Path: {subset_path}")
            if hasattr(dataset, 'num_rows'):
                print(f"     Rows: {dataset.num_rows}")
            elif 'train' in dataset:
                print(f"     Rows (train): {len(dataset['train'])}")
            
        except Exception as e:
            print(f"\n[FAIL] Error downloading {subset}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UltraDomain subsets")
    parser.add_argument(
        "--output-dir",
        default="data/ultradomain_hf",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["agriculture", "legal", "cs", "mix"],
        choices=["agriculture", "legal", "cs", "mix", "mathematics", "physics", "politics"],
        help="Subsets to download"
    )
    
    args = parser.parse_args()
    download_ultradomain_subsets(args.output_dir, args.subsets)
