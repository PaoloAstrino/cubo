#!/usr/bin/env python3
"""
Download individual BEIR datasets.

Usage:
    python scripts/download_beir_dataset.py scifact
    python scripts/download_beir_dataset.py --all
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

BEIR_DATASETS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "trec-covid": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
    "webis-touche2020": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "dbpedia-entity": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
    "fever": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip",
    "climate-fever": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip",
    "nq": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
    "hotpotqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
}


def download_file(url: str, dest_path: Path) -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip file."""
    try:
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✓ Extracted to: {extract_dir}")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def download_dataset(dataset_name: str, output_dir: Path = Path("data/beir")) -> bool:
    """Download and extract a BEIR dataset."""
    if dataset_name not in BEIR_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {', '.join(BEIR_DATASETS.keys())}")
        return False
    
    url = BEIR_DATASETS[dataset_name]
    dataset_dir = output_dir / dataset_name
    
    # Check if already exists
    if dataset_dir.exists() and (dataset_dir / "corpus.jsonl").exists():
        print(f"Dataset '{dataset_name}' already exists at {dataset_dir}")
        return True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{dataset_name}.zip"
    
    # Download
    print(f"Downloading {dataset_name}...")
    if not download_file(url, zip_path):
        return False
    
    # Extract
    if not extract_zip(zip_path, output_dir):
        return False
    
    # Clean up zip
    zip_path.unlink()
    print(f"✓ Dataset '{dataset_name}' ready at {dataset_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download BEIR datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name to download (or --all for all datasets)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/beir",
        help="Output directory for datasets",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available BEIR datasets:")
        for name in sorted(BEIR_DATASETS.keys()):
            print(f"  - {name}")
        return
    
    output_dir = Path(args.output_dir)
    
    if args.all:
        print(f"Downloading all {len(BEIR_DATASETS)} BEIR datasets...")
        success_count = 0
        for dataset_name in BEIR_DATASETS.keys():
            if download_dataset(dataset_name, output_dir):
                success_count += 1
        print(f"\n✓ Successfully downloaded {success_count}/{len(BEIR_DATASETS)} datasets")
    elif args.dataset:
        download_dataset(args.dataset, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
