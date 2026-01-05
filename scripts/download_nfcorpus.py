import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_nfcorpus(output_dir="data/beir/nfcorpus"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "nfcorpus.zip"

    print(f"Downloading NFCorpus from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

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

    print(f"Data ready at data/beir/nfcorpus")


if __name__ == "__main__":
    download_nfcorpus()
