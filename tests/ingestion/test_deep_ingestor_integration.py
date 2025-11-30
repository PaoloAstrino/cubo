import json
from pathlib import Path

import pandas as pd

from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.ingestion.fast_pass_ingestor import FastPassIngestor


def test_fast_then_deep_compatibility(tmp_path: Path):
    # Create input files
    folder = tmp_path / "docs"
    folder.mkdir()
    file1 = folder / "a.txt"
    file2 = folder / "b.txt"
    file1.write_text("A short document about compatibility. This will be chunked.")
    file2.write_text("Another doc. Will also be chunked similarly.")

    fast_out = tmp_path / "fast_out"
    deep_out = tmp_path / "deep_out"

    # Run fast pass which saves JSONL
    fp = FastPassIngestor(output_dir=str(fast_out), skip_model=True)
    fp.ingest_folder(str(folder))

    # Run deep pass
    dp = DeepIngestor(input_folder=str(folder), output_dir=str(deep_out))
    result = dp.ingest()

    # Read both results
    fast_jsonl = fast_out / "chunks.jsonl"
    assert fast_jsonl.exists()

    deep_parquet = Path(result["chunks_parquet"])
    assert deep_parquet.exists()
    deep_df = pd.read_parquet(deep_parquet)

    # Ensure some stability exists: every file_hash in deep df exists on fast jsonl
    fast_hashes = set()
    with open(fast_jsonl, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            fast_hashes.add(rec.get("file_hash"))

    assert fast_hashes.issuperset(set(deep_df["file_hash"].unique()))
