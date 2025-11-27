import math
from pathlib import Path

import pandas as pd

from src.cubo.ingestion.deep_ingestor import DeepIngestor


def _write_csv(path: Path, rows: int) -> None:
    header = "id,value"
    lines = [header]
    for i in range(rows):
        lines.append(f"{i},value_{i}")
    path.write_text("\n".join(lines))


def test_deep_ingestor_creates_chunks_parquet(tmp_path: Path):
    folder = tmp_path / "docs"
    folder.mkdir()
    doc = folder / "story.txt"
    doc.write_text("Deep chunking test. Won't be huge.")

    output = tmp_path / "deep_out"
    ingestor = DeepIngestor(input_folder=str(folder), output_dir=str(output))
    result = ingestor.ingest()

    parquet_path = Path(result["chunks_parquet"])
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert not df.empty
    assert "chunk_id" in df.columns
    assert all(df["file_hash"].notna())


def test_chunk_id_stability_for_identical_files(tmp_path: Path):
    content = "Stable chunking content. Short enough for a single chunk."

    base = tmp_path / "docs"
    first = base / "run1"
    first.mkdir(parents=True)
    (first / "content.txt").write_text(content)

    second = base / "run2"
    second.mkdir(parents=True)
    (second / "other.txt").write_text(content)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    df1 = pd.read_parquet(
        DeepIngestor(input_folder=str(first), output_dir=str(out1)).ingest()["chunks_parquet"]
    )
    df2 = pd.read_parquet(
        DeepIngestor(input_folder=str(second), output_dir=str(out2)).ingest()["chunks_parquet"]
    )

    assert list(df1["chunk_id"]) == list(df2["chunk_id"])


def test_csv_chunking_groups_rows(tmp_path: Path):
    folder = tmp_path / "csv_docs"
    folder.mkdir()
    csv_path = folder / "table.csv"
    _write_csv(csv_path, 25)

    output = tmp_path / "csv_out"
    ingestor = DeepIngestor(
        input_folder=str(folder),
        output_dir=str(output),
        csv_rows_per_chunk=10,
    )
    result = ingestor.ingest()

    metadata = pd.read_parquet(result["chunks_parquet"])
    assert len(metadata) == math.ceil(25 / 10)
    assert metadata["chunk_id"].str.contains("_csv_").all()


def test_deep_ingestor_resume(tmp_path: Path):
    folder = tmp_path / "docs_resume"
    folder.mkdir()
    (folder / "a.txt").write_text("This is a test file A.")

    output = tmp_path / "deep_out"
    ingestor = DeepIngestor(input_folder=str(folder), output_dir=str(output))
    res1 = ingestor.ingest()
    parquet1 = res1["chunks_parquet"]
    assert parquet1
    df1 = pd.read_parquet(parquet1)

    # Add a new file and resume
    (folder / "b.txt").write_text("This is a test file B with different content.")
    res2 = DeepIngestor(input_folder=str(folder), output_dir=str(output)).ingest(resume=True)
    parquet2 = res2.get("chunks_parquet")
    # New parquet should exist and only contain new files
    assert parquet2
    df2 = pd.read_parquet(parquet2)
    assert all(fp not in list(df1["file_path"]) for fp in df2["file_path"])
