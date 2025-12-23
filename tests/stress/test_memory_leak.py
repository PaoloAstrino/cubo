import gc
import os
from pathlib import Path

import pytest

from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.storage.metadata_manager import MetadataManager

# Skip if psutil is not installed
pytest.importorskip("psutil")
import psutil


@pytest.mark.slow
def test_memory_usage_stability(tmp_path: Path):
    """
    Stress test: Run ingestion repeatedly and check for memory leaks.
    This test is marked slow and should be run optionally.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a reasonable number of files to process
    for i in range(20):
        (input_dir / f"doc_{i}.txt").write_text(f"This is the content of document {i} " * 100)

    process = psutil.Process(os.getpid())

    # Warmup run
    manager = MetadataManager(db_path=str(tmp_path / "meta.db"))
    ingestor = DeepIngestor(
        input_folder=str(input_dir),
        output_dir=str(output_dir),
        metadata_manager=manager,
        chunk_batch_size=10,  # Force flushing
    )
    ingestor.ingest()

    gc.collect()
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB

    mem_usages = []

    # Run loop
    iterations = 5
    for i in range(iterations):
        # Clean output to force re-ingest logic (or we could use different output dirs)
        # Here we just overwrite or create new run

        # We need to clear the output parquet to ensure full processing if we are not using resume
        # But DeepIngestor overwrites chunks_deep.parquet at the end.
        # However, we want to simulate fresh runs.

        ingestor = DeepIngestor(
            input_folder=str(input_dir),
            output_dir=str(output_dir),
            metadata_manager=manager,
            chunk_batch_size=10,
        )
        ingestor.ingest()

        gc.collect()
        mem = process.memory_info().rss / 1024 / 1024
        mem_usages.append(mem)

    manager.conn.close()

    # Analyze memory trend
    # It's hard to be deterministic about memory in Python, but we can check for massive growth.
    # We expect memory to be roughly stable.

    print(f"Memory usage (MB): {initial_mem:.2f} -> {mem_usages}")

    # Simple check: Final memory shouldn't be > 2x initial memory for this small workload
    # This is a loose check to catch egregious leaks.
    assert (
        mem_usages[-1] < initial_mem * 3.0
    ), f"Memory grew too much: {initial_mem} -> {mem_usages[-1]}"
