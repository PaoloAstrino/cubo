import os
from pathlib import Path

import pytest

# Skip the test gracefully if Hypothesis is not installed in the environment.
pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.storage.metadata_manager import MetadataManager


@settings(max_examples=5, deadline=None)
@given(bad_bytes=st.binary(min_size=1, max_size=2048))
def test_corrupted_pdf_is_handled_without_crash(tmp_path: Path, bad_bytes: bytes):
    """Random PDF-like bytes should not crash ingestion and should produce a file status entry."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    bad_pdf = input_dir / "bad.pdf"
    bad_pdf.write_bytes(bad_bytes)

    manager = MetadataManager(db_path=str(tmp_path / "meta.db"))
    run_id = "run-corrupt"

    ingestor = DeepIngestor(
        input_folder=str(input_dir),
        output_dir=str(output_dir),
        run_id=run_id,
        metadata_manager=manager,
    )

    # Should not raise even if PDF is corrupt
    res = ingestor.ingest()
    assert res is not None

    status = manager.get_file_status(run_id, str(bad_pdf))
    assert status is not None
    # Accept either failed (expected for corrupt) or succeeded (if fallback parser can read it), but never crash
    assert status["status"] in {"failed", "succeeded", "processing"}

    manager.conn.close()
