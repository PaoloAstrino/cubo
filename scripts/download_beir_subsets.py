#!/usr/bin/env python3
"""
Download BeIR Subsets

Downloads FiQA and NFCorpus subsets for privacy/health domains.
Total size: ~3-4 GB

Usage:
    python scripts/download_beir_subsets.py --output-dir data/beir_hf
"""

import argparse
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Install with: pip install datasets")
    exit(1)


def download_beir_subsets(output_dir: str, subsets: list = None):
    """Download BeIR subsets from HuggingFace.
    
    Args:
        output_dir: Directory to save the datasets
        subsets: List of subsets to download (fiqa, nfcorpus)
    """
    if subsets is None:
        subsets = ["fiqa", "nfcorpus"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading BeIR subsets from HuggingFace...")
    print(f"Subsets: {', '.join(subsets)}")
    print("This may take 5-10 minutes...\n")
    
    for subset in subsets:
        try:
            print(f"\n{'='*60}")
            print(f"Downloading: {subset}")
            print(f"{'='*60}")
            
            # Load dataset - BeIR uses different repo structure
            dataset = load_dataset(f"BeIR/{subset}", "corpus")
            
            # Save to disk
            subset_path = output_path / f"beir_{subset}"
            dataset.save_to_disk(str(subset_path))
            
            # Print info
            print(f"\n[OK] Downloaded {subset}")
            print(f"     Path: {subset_path}")
            if hasattr(dataset, 'num_rows'):
                print(f"     Rows: {dataset.num_rows}")
            elif 'corpus' in dataset:
                print(f"     Rows (corpus): {len(dataset['corpus'])}")
            
            # Also download queries if available
            try:
                queries = load_dataset(f"BeIR/{subset}", "queries")
                queries_path = subset_path.parent / f"beir_{subset}_queries"
                queries.save_to_disk(str(queries_path))
                print(f"     Queries: {len(queries)} (saved to {queries_path.name})")
            except:
                print(f"     Queries: Not available separately")
                
        except Exception as e:
            print(f"\n[FAIL] Error downloading {subset}: {e}")
            print(f"     Trying alternative approach...")
            
            # Alternative: try without split specification
            try:
                dataset = load_dataset(f"BeIR/{subset}")
                subset_path = output_path / f"beir_{subset}"
                dataset.save_to_disk(str(subset_path))
                print(f"[OK] Downloaded {subset} (alternative method)")
            except Exception as e2:
                print(f"[FAIL] Alternative also failed: {e2}")
                continue
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BeIR subsets")
    parser.add_argument(
        "--output-dir",
        default="data/beir_hf",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["fiqa", "nfcorpus"],
        choices=["fiqa", "nfcorpus", "scifact", "trec-covid", "nq"],
        help="Subsets to download"
    )
    
    args = parser.parse_args()
    download_beir_subsets(args.output_dir, args.subsets)
