"""
NFCorpus Dataset Downloader

This script downloads and extracts the NFCorpus dataset from the BEIR benchmark collection.
NFCorpus is a dataset for non-factoid question answering in the biomedical domain.

The script:
1. Downloads the dataset zip file from the BEIR repository
2. Extracts it to data/beir/nfcorpus
3. Provides progress feedback during download
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_nfcorpus(output_dir="data/beir/nfcorpus"):
    """Download and extract the NFCorpus dataset.

    Args:
        output_dir: Directory where to extract the dataset (default: data/beir/nfcorpus)
    """
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "nfcorpus.zip"

    print(f"Downloading NFCorpus from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress bar
    with open(zip_path, "wb") as f, tqdm(
        desc="nfcorpus.zip",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data/beir")

    print("Data ready at data/beir/nfcorpus")


if __name__ == "__main__":
    download_nfcorpus()
