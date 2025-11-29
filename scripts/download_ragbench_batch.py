#!/usr/bin/env python3
"""
Batch Download RAGBench Subsets

Downloads multiple RAGBench configs in one command.
Recommended: pubmedqa, finqa, hotpotqa (~4-5 GB total)

Usage:
    python scripts/download_ragbench_batch.py --configs pubmedqa finqa hotpotqa --output-dir data/ragbench
"""

import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Install with: pip install datasets")
    sys.exit(1)


def download_ragbench_batch(output_dir: str, configs: list, split: str = "test"):
    """Download multiple RAGBench configs.
    
    Args:
        output_dir: Directory to save datasets
        configs: List of config names to download
        split: Dataset split (train, validation, test)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(configs)} RAGBench configs from HuggingFace...")
    print(f"Configs: {', '.join(configs)}")
    print(f"Split: {split}")
    print("This may take 10-15 minutes...\n")
    
    results = {"success": [], "failed": []}
    
    for config in configs:
        try:
            print(f"\n{'='*60}")
            print(f"Downloading: {config} ({split})")
            print(f"{'='*60}")
            
            # Load dataset
            dataset = load_dataset("rungalileo/ragbench", config, split=split)
            
            # Save as parquet
            output_file = output_path / f"ragbench_{config}_{split}.parquet"
            dataset.to_parquet(str(output_file))
            
            print(f"\n[OK] Downloaded {config}")
            print(f"     File: {output_file.name}")
            print(f"     Rows: {len(dataset)}")
            print(f"     Columns: {', '.join(dataset.column_names[:5])}...")
            
            results["success"].append(config)
            
        except Exception as e:
            print(f"\n[FAIL] Error downloading {config}: {e}")
            results["failed"].append(config)
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(results['success'])}/{len(configs)}")
    for cfg in results['success']:
        print(f"  [OK] {cfg}")
    
    if results['failed']:
        print(f"\nFailed: {len(results['failed'])}")
        for cfg in results['failed']:
            print(f"  [FAIL] {cfg}")
    
    print(f"\nOutput directory: {output_path}")
    print(f"{'='*60}\n")
    
    return len(results['failed']) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download RAGBench subsets")
    parser.add_argument(
        "--output-dir",
        default="data/ragbench",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["pubmedqa", "finqa", "hotpotqa"],
        choices=["covidqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa", "tatqa", "techqa",
                 "cuad", "delucionqa", "emanual", "expertqa", "finqa"],
        help="Configs to download (space-separated)"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split"
    )
    
    args = parser.parse_args()
    success = download_ragbench_batch(args.output_dir, args.configs, args.split)
    
    sys.exit(0 if success else 1)
