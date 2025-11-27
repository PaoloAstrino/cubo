#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cubo.ingestion.deep_ingestor import build_deep_index


def test_deep_ingest_parquet_and_chunk_id_stability(tmp_path):
    sample_folder = Path(__file__).parent.parent / "data"
    assert sample_folder.exists(), "Data folder not found for test"

    out_dir = tmp_path / "deep_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result1 = build_deep_index(str(sample_folder), output_dir=str(out_dir), skip_model=True)
    assert result1 and "parquet" in result1 and result1["chunks_count"] > 0

    df1 = pd.read_parquet(result1["parquet"])
    assert "chunk_id" in df1.columns
    ids1 = set(df1["chunk_id"].astype(str).tolist())

    # Run again to check stability
    result2 = build_deep_index(str(sample_folder), output_dir=str(out_dir), skip_model=True)
    assert result2 and "parquet" in result2 and result2["chunks_count"] > 0
    df2 = pd.read_parquet(result2["parquet"])
    ids2 = set(df2["chunk_id"].astype(str).tolist())

    # Chunk IDs should be identical sets across runs
    assert ids1 == ids2
