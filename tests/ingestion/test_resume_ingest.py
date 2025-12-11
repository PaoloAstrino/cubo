import pandas as pd
import pytest
from pathlib import Path
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.storage.metadata_manager import MetadataManager

def test_resume_ingestion_skips_processed_files(tmp_path: Path):
    """Test that resume=True skips files already present in the existing parquet."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    # Create two files
    file1 = input_dir / "doc1.txt"
    file1.write_text("Content of document 1")
    file2 = input_dir / "doc2.txt"
    file2.write_text("Content of document 2")
    
    # Simulate a previous run that processed doc1
    # We create a parquet file with chunks for doc1
    existing_data = [
        {
            "chunk_id": "chunk1",
            "text": "Content of document 1",
            "file_path": str(file1),
            "filename": "doc1.txt",
            "chunk_index": 0
        }
    ]
    df = pd.DataFrame(existing_data)
    df.to_parquet(output_dir / "chunks_deep.parquet", index=False)
    
    manager = MetadataManager(db_path=str(tmp_path / "meta.db"))
    
    ingestor = DeepIngestor(
        input_folder=str(input_dir),
        output_dir=str(output_dir),
        metadata_manager=manager
    )
    
    # Run ingest with resume=True
    result = ingestor.ingest(resume=True)
    
    # Verify results
    processed_files = result["processed_files"]
    
    # doc1 should be skipped, doc2 should be processed
    assert str(file1) not in processed_files
    assert str(file2) in processed_files
    
    # Verify final parquet contains both
    final_df = pd.read_parquet(result["chunks_parquet"])
    assert len(final_df) == 2
    assert str(file1) in final_df["file_path"].values
    assert str(file2) in final_df["file_path"].values
    
    manager.conn.close()

def test_resume_ingestion_no_existing_parquet(tmp_path: Path):
    """Test that resume=True works fine even if no previous parquet exists."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    file1 = input_dir / "doc1.txt"
    file1.write_text("Content 1")
    
    manager = MetadataManager(db_path=str(tmp_path / "meta.db"))
    
    ingestor = DeepIngestor(
        input_folder=str(input_dir),
        output_dir=str(output_dir),
        metadata_manager=manager
    )
    
    result = ingestor.ingest(resume=True)
    
    assert str(file1) in result["processed_files"]
    assert Path(result["chunks_parquet"]).exists()
    
    manager.conn.close()
