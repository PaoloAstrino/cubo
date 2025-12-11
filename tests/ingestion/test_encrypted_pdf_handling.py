import pytest
from pathlib import Path
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.storage.metadata_manager import MetadataManager

# Skip if Hypothesis is not installed
pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st, HealthCheck

@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(content=st.binary(min_size=100, max_size=5000))
def test_encrypted_pdf_handling(tmp_path: Path, content: bytes):
    """Fuzz test: Encrypted or malformed PDF bytes should not crash ingestion."""
    # Create input directory and file
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Inject PDF header and encryption marker to simulate encrypted PDF
    # This makes it more likely to trigger PDF parsing logic than random bytes
    pdf_content = b"%PDF-1.5\n" + b"/Encrypt" + content + b"%%EOF"
    
    pdf_file = input_dir / "locked.pdf"
    pdf_file.write_bytes(pdf_content)
    
    manager = MetadataManager(db_path=str(tmp_path / "meta.db"))
    run_id = "run-fuzz-enc"
    
    ingestor = DeepIngestor(
        input_folder=str(input_dir),
        output_dir=str(output_dir),
        run_id=run_id,
        metadata_manager=manager
    )
    
    # Should not raise exception
    try:
        ingestor.ingest()
    except Exception as e:
        pytest.fail(f"Ingestion crashed on encrypted/fuzzed PDF: {e}")
        
    # Verify status was recorded
    status = manager.get_file_status(run_id, str(pdf_file))
    assert status is not None
    # It might fail (due to encryption) or succeed (if fallback works), but must be recorded
    assert status["status"] in ("failed", "succeeded", "processing")
    
    manager.conn.close()
