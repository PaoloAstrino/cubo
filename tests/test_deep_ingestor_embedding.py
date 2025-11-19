from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock, patch

from sentence_transformers import SentenceTransformer
from src.cubo.ingestion.deep_ingestor import DeepIngestor
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.config import config

try:
    import chromadb.config
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

import pytest


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_deep_chunks_can_be_embedded_and_inserted(tmp_path: Path):
    with patch('src.cubo.config.config', {"vector_store_backend": "chroma"}):
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "a.txt").write_text("This is an embedding test document. It has sentences to chunk.")

        out = tmp_path / "out"
        res = DeepIngestor(input_folder=str(folder), output_dir=str(out)).ingest()
        df = pd.read_parquet(res['chunks_parquet'])
        texts = df['text'].tolist()
        metadatas = df.to_dict(orient='records')
        chunk_ids = df['chunk_id'].tolist()

        # Mock a small SentenceTransformer model that returns 64-d vectors
        mock_model = MagicMock(spec=SentenceTransformer)
        def mock_encode(texts, batch_size=1):
            # Return deterministic embeddings by text length
            return [[float(len(t.split()))] * 64 for t in texts]
        mock_model.encode.side_effect = mock_encode

        # Use retriever with mock model
        retriever = DocumentRetriever(model=mock_model)
        # Use a dedicated collection name for this test
        retriever.collection = retriever.client.get_or_create_collection("test_deep_embeddings")

        # Generate embeddings using the mocked model directly
        embeddings = mock_model.encode(texts, batch_size=32)
        # Insert into collection
        retriever._add_chunks_to_collection(embeddings, texts, metadatas, chunk_ids, "test_doc.txt")

        # Confirm that the collection has the inserted count
        assert retriever.collection.count() == len(texts)
