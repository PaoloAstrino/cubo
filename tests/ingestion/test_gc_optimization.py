from unittest.mock import patch

from cubo.ingestion.deep_ingestor import DeepIngestor


def test_explicit_gc_on_flush(tmp_path):
    """
    Verify that gc.collect() is called after flushing a batch of chunks.
    This ensures that memory is actively reclaimed during heavy ingestion.
    """
    output_dir = tmp_path / "deep_output"
    output_dir.mkdir()

    ingestor = DeepIngestor(input_folder=str(tmp_path), output_dir=str(output_dir))

    chunks = [{"text": "chunk1", "chunk_id": "1"}]

    # Mock gc.collect to verify it gets called
    with patch("gc.collect") as mock_gc:
        ingestor._flush_chunk_batch(chunks)

        # Verify parquet file was created
        assert len(list(output_dir.glob("*.parquet"))) == 1

        # Verify GC was triggered
        mock_gc.assert_called_once()
        print("Verified: gc.collect() called after flush.")
