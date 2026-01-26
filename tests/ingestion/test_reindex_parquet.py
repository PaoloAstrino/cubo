import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from cubo.config import config
from cubo.ingestion.deep_ingestor import DeepIngestor


def test_reindex_parquet_with_wipe(tmp_path: Path):
    # Create input files to produce a parquet
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Test doc for reindex")

    out = tmp_path / "deep"
    ing = DeepIngestor(input_folder=str(folder), output_dir=str(out))
    _res = ing.ingest()
    parquet = _res["chunks_parquet"]
    df = pd.read_parquet(parquet)

    # Setup temp FAISS index path
    tmpdb = tmp_path / "faiss"
    tmpdb.mkdir()
    config.set("vector_store_path", str(tmpdb))
    config.set("vector_store_backend", "faiss")

    # Ensure model loading is mocked to avoid heavy dependencies
    import runpy

    # Run reindex script in-process with model patched
    with patch("cubo.embeddings.model_loader.model_manager.get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 64

        def mock_encode(texts, batch_size=1):
            return [[0.1] * 64 for _ in texts]

        mock_model.encode.side_effect = mock_encode
        mock_get_model.return_value = mock_model
        script = str(Path.cwd() / "scripts" / "reindex_parquet.py")
        # set CLI args and execute script in-process
        old_argv = sys.argv
        try:
            sys.argv = [
                "reindex_parquet.py",
                "--parquet",
                parquet,
                "--collection",
                "test_reindex",
                "--replace-collection",
                "--wipe-db",
            ]
            runpy.run_path(script, run_name="__main__")
        finally:
            sys.argv = old_argv

    # Verify collection count
    from cubo.retrieval.retriever import DocumentRetriever

    retr = DocumentRetriever(model=None)
    coll = retr.collection
    assert coll.count() == len(df)
    # Close retriever explicitly to release DB handles and background threads
    close_fn = getattr(retr, "close", None)
    if callable(close_fn):
        close_fn()
