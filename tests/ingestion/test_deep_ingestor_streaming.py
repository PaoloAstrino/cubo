"""Tests for streaming saves in DeepIngestor.

These tests verify that the DeepIngestor properly flushes chunks
to temporary parquet files during ingestion to prevent RAM accumulation.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cubo.ingestion.deep_ingestor import DeepIngestor


class TestStreamingSaves:
    """Tests for the streaming save functionality."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for input and output."""
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                yield Path(input_dir), Path(output_dir)

    def _create_test_files(self, folder: Path, num_files: int, lines_per_file: int = 50):
        """Create test text files."""
        for i in range(num_files):
            content = "\n".join([f"Line {j} of file {i}" for j in range(lines_per_file)])
            (folder / f"doc_{i}.txt").write_text(content)

    def test_streaming_save_creates_final_parquet(self, temp_dirs):
        """Test that streaming saves produce correct final parquet."""
        input_dir, output_dir = temp_dirs

        # Create multiple files to generate enough chunks
        self._create_test_files(input_dir, num_files=10)

        ingestor = DeepIngestor(
            input_folder=str(input_dir),
            output_dir=str(output_dir),
            chunk_batch_size=5,  # Small batch to trigger flushes
        )

        result = ingestor.ingest()

        assert result["chunks_count"] > 0
        assert Path(result["chunks_parquet"]).exists()

        # Verify parquet is readable
        df = pd.read_parquet(result["chunks_parquet"])
        assert len(df) == result["chunks_count"]

    def test_temp_files_are_cleaned_up(self, temp_dirs):
        """Test that temporary parquet files are cleaned up after merge."""
        input_dir, output_dir = temp_dirs

        self._create_test_files(input_dir, num_files=10)

        ingestor = DeepIngestor(
            input_folder=str(input_dir),
            output_dir=str(output_dir),
            chunk_batch_size=3,  # Very small to ensure multiple temp files
        )

        result = ingestor.ingest()

        # Check no temp files remain
        temp_files = list(output_dir.glob("temp_chunks_*.parquet"))
        assert len(temp_files) == 0

        # But final file should exist
        assert Path(result["chunks_parquet"]).exists()

    def test_chunk_batch_size_from_config(self, temp_dirs):
        """Test that chunk_batch_size can be configured."""
        input_dir, output_dir = temp_dirs

        self._create_test_files(input_dir, num_files=5)

        # Custom batch size
        ingestor = DeepIngestor(
            input_folder=str(input_dir), output_dir=str(output_dir), chunk_batch_size=100
        )

        assert ingestor.chunk_batch_size == 100

    def test_resume_still_works(self, temp_dirs):
        """Test that resume functionality still works with streaming saves."""
        input_dir, output_dir = temp_dirs

        # First run - create initial files
        self._create_test_files(input_dir, num_files=3)

        ingestor1 = DeepIngestor(
            input_folder=str(input_dir), output_dir=str(output_dir), chunk_batch_size=5
        )
        result1 = ingestor1.ingest()
        result1["chunks_count"]

        # Add more files
        for i in range(3, 6):
            content = f"New content for file {i}"
            (input_dir / f"doc_{i}.txt").write_text(content)

        # Resume ingestion
        ingestor2 = DeepIngestor(
            input_folder=str(input_dir), output_dir=str(output_dir), chunk_batch_size=5
        )
        result2 = ingestor2.ingest(resume=True)

        # Should have processed new files
        assert result2["chunks_count"] > 0
        assert (
            "doc_3.txt" in str(result2["processed_files"])
            or "doc_4.txt" in str(result2["processed_files"])
            or "doc_5.txt" in str(result2["processed_files"])
        )

    def test_empty_input_handled(self, temp_dirs):
        """Test handling of empty input folder."""
        input_dir, output_dir = temp_dirs

        ingestor = DeepIngestor(
            input_folder=str(input_dir), output_dir=str(output_dir), chunk_batch_size=5
        )

        result = ingestor.ingest()
        assert result == {}

    def test_single_small_file(self, temp_dirs):
        """Test with a single small file (no temp files needed)."""
        input_dir, output_dir = temp_dirs

        # Create just one small file
        (input_dir / "small.txt").write_text("Small content.")

        ingestor = DeepIngestor(
            input_folder=str(input_dir),
            output_dir=str(output_dir),
            chunk_batch_size=100,  # Large batch - won't trigger flush
        )

        result = ingestor.ingest()

        assert result["chunks_count"] >= 1
        assert Path(result["chunks_parquet"]).exists()


class TestDeepIngestorBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as input_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                yield Path(input_dir), Path(output_dir)

    def test_result_structure_unchanged(self, temp_dirs):
        """Test that result dictionary structure is unchanged."""
        input_dir, output_dir = temp_dirs

        (input_dir / "test.txt").write_text("Test content for backward compatibility.")

        ingestor = DeepIngestor(input_folder=str(input_dir), output_dir=str(output_dir))

        result = ingestor.ingest()

        # Verify expected keys
        assert "chunks_parquet" in result
        assert "manifest" in result
        assert "chunks_count" in result
        assert "processed_files" in result

    def test_manifest_created(self, temp_dirs):
        """Test that manifest file is still created."""
        input_dir, output_dir = temp_dirs

        (input_dir / "test.txt").write_text("Content.")

        ingestor = DeepIngestor(input_folder=str(input_dir), output_dir=str(output_dir))

        result = ingestor.ingest()

        assert Path(result["manifest"]).exists()

    def test_chunk_ids_stable(self, temp_dirs):
        """Test that chunk IDs remain stable across runs."""
        input_dir, output_dir = temp_dirs

        content = "Stable content for ID generation."
        (input_dir / "stable.txt").write_text(content)

        # First run
        ingestor1 = DeepIngestor(input_folder=str(input_dir), output_dir=str(output_dir))
        result1 = ingestor1.ingest()
        df1 = pd.read_parquet(result1["chunks_parquet"])

        # Reset output
        os.remove(result1["chunks_parquet"])
        os.remove(result1["manifest"])

        # Second run
        ingestor2 = DeepIngestor(input_folder=str(input_dir), output_dir=str(output_dir))
        result2 = ingestor2.ingest()
        df2 = pd.read_parquet(result2["chunks_parquet"])

        # Chunk IDs should be identical
        assert list(df1["chunk_id"]) == list(df2["chunk_id"])
