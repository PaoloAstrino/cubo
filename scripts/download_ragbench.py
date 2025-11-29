#!/usr/bin/env python3
"""
Download RAGBench Dataset

Downloads RAGBench from HuggingFace and saves to local directory.
Dataset: https://huggingface.co/datasets/rungalileo/ragbench

RAGBench consists of 12 sub-component datasets across 5 domains:
- Biomedical, Legal, Customer Support, Finance, General Knowledge
"""

import argparse
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' library not installed. Install with: pip install datasets")
    exit(1)


def download_ragbench(output_dir: str, config: str = "covidqa", split: str = "test", num_samples: int = None):
    """Download RAGBench dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        config: Sub-dataset name (covidqa, hagrid, hotpotqa, msmarco, pubmedqa, tatqa, techqa)
        split: Dataset split to download (train, validation, test)
        num_samples: If set, only download this many samples (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading RAGBench - {config} ({split} split) from HuggingFace...")
    print("This may take a few minutes...")
    
    try:
        # Load the dataset with config name
        if num_samples:
            # Use streaming for sampling
            dataset = load_dataset(
                "rungalileo/ragbench", 
                config,
                split=split,
                streaming=True
            )
            # Take only num_samples
            dataset = dataset.take(num_samples)
            # Convert to regular dataset
            from datasets import Dataset
            dataset = Dataset.from_generator(lambda: dataset)
        else:
            dataset = load_dataset("rungalileo/ragbench", config, split=split)
        
        # Save to disk in Parquet format
        output_file = output_path / f"ragbench_{config}_{split}.parquet"
        dataset.to_parquet(str(output_file))
        
        print(f"✓ Downloaded {len(dataset)} examples to {output_file}")
        print(f"\nDataset columns: {dataset.column_names}")
        
        # Print sample
        if len(dataset) > 0:
            print(f"\nSample row:")
            for key, value in dataset[0].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        return output_file
        
    except Exception as e:
        print(f"ERROR downloading RAGBench: {e}")
        print("\nMake sure you have:")
        print("  1. Installed 'datasets' library: pip install datasets")
        print("  2. Internet connection")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download RAGBench dataset")
    parser.add_argument(
        "--output-dir",
        default="data/ragbench",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--config",
        default="covidqa",
        choices=["covidqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa", "tatqa", "techqa"],
        help="Sub-dataset to download"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to download (for testing)"
    )
    
    args = parser.parse_args()
    download_ragbench(args.output_dir, args.config, args.split, args.num_samples)
