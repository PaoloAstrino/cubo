import pytest

pytest.importorskip("torch")

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from sentence_transformers import SentenceTransformer

from cubo.config import config
from cubo.ingestion.deep_ingestor import DeepIngestor
from cubo.retrieval.retriever import DocumentRetriever


def test_deep_chunks_can_be_embedded_and_inserted(tmp_path: Path):
    # Temporarily override vector store backend to FAISS for the test
    orig_backend = config.get("vector_store_backend", None)
    orig_path = config.get("vector_store_path", None)
    config.set("vector_store_backend", "faiss")
    # Use separate index dir for isolation
    tmpdb = tmp_path / "faiss"
    tmpdb.mkdir()
    config.set("vector_store_path", str(tmpdb))
    try:
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "a.txt").write_text(
            "This is an embedding test document. It has sentences to chunk."
        )

        out = tmp_path / "out"
        res = DeepIngestor(input_folder=str(folder), output_dir=str(out)).ingest()
        df = pd.read_parquet(res["chunks_parquet"])
        texts = df["text"].tolist()
        metadatas = df.to_dict(orient="records")
        chunk_ids = df["chunk_id"].tolist()

        # Mock a small SentenceTransformer model that returns 64-d vectors
        mock_model = MagicMock(spec=SentenceTransformer)

        def mock_encode(texts, batch_size=1):
            # Return deterministic embeddings by text length
            return [[float(len(t.split()))] * 64 for t in texts]

        mock_model.encode.side_effect = mock_encode
        # Ensure the model reports an embedding dimension (used by FAISS)
        mock_model.get_sentence_embedding_dimension.return_value = 64

        # Use retriever with mock model (FAISS backend)
        retriever = DocumentRetriever(model=mock_model)
        # Ensure a clean collection for this test
        retriever.collection.reset()

        # Generate embeddings using the mocked model directly
        embeddings = mock_model.encode(texts, batch_size=32)
        # Insert into collection
        retriever._add_chunks_to_collection(embeddings, texts, metadatas, chunk_ids, "test_doc.txt")

        # Confirm that the collection has the inserted count
        assert retriever.collection.count() == len(texts)
    finally:
        # restore config
        config.set("vector_store_backend", orig_backend)
        if orig_path is not None:
            config.set("vector_store_path", orig_path)
