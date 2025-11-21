from pathlib import Path
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pandas as pd
from src.cubo.ingestion.deep_ingestor import DeepIngestor
from src.cubo.config import config

try:
    import chromadb.config
    CHROMADB_AVAILABLE = True
except Exception:
    # chromadb may import but fail during initialization if optional dependencies (e.g., onnxruntime) are missing
    CHROMADB_AVAILABLE = False

import pytest


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_reindex_parquet_with_wipe(tmp_path: Path):
    # Create input files to produce a parquet
    folder = tmp_path / 'docs'
    folder.mkdir()
    (folder / 'a.txt').write_text('Test doc for reindex')

    out = tmp_path / 'deep'
    ing = DeepIngestor(input_folder=str(folder), output_dir=str(out))
    res = ing.ingest()
    parquet = res['chunks_parquet']
    df = pd.read_parquet(parquet)

    # Setup temp chroma db path
    tmpdb = tmp_path / 'chroma'
    tmpdb.mkdir()
    config.set('chroma_db_path', str(tmpdb))
    config.set('vector_store_backend', 'chroma')

    # Ensure model loading is mocked to avoid heavy dependencies
    import runpy
    import sys
    # Run reindex script in-process with model patched
    with patch('src.cubo.embeddings.model_loader.model_manager.get_model') as mock_get_model:
        mock_model = MagicMock()
        def mock_encode(texts, batch_size=1):
            return [[0.1] * 64 for _ in texts]
        mock_model.encode.side_effect = mock_encode
        mock_get_model.return_value = mock_model
        script = str(Path.cwd() / 'scripts' / 'reindex_parquet.py')
        # set CLI args and execute script in-process
        old_argv = sys.argv
        try:
            sys.argv = ['reindex_parquet.py', '--parquet', parquet, '--collection', 'test_reindex', '--replace-collection', '--wipe-db']
            runpy.run_path(script, run_name='__main__')
        finally:
            sys.argv = old_argv

    # Verify collection count
    from src.cubo.retrieval.retriever import DocumentRetriever
    retr = DocumentRetriever(model=None)
    coll = retr.client.get_or_create_collection('test_reindex')
    assert coll.count() == len(df)
