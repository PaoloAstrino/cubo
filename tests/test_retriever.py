import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer
from src.retriever import DocumentRetriever
from src.config import config
import chromadb

@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model."""
    model = MagicMock(spec=SentenceTransformer)
    # Mock encode to return a tensor-like object with tolist
    mock_tensor = MagicMock()
    mock_tensor.tolist.return_value = [[0.1] * 768]
    model.encode.return_value = mock_tensor
    return model

@pytest.fixture
def temp_db_path():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_add_documents_empty(mock_model, temp_db_path):
    """Test adding empty documents list."""
    with patch('src.config.config', {"vector_db_path": temp_db_path, "embedding_batch_size": 32}):
        # Use a unique collection name to avoid conflicts
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_empty")
        retriever.add_documents([])  # Should not crash
        assert retriever.collection.count() == 0

def test_add_documents_success(mock_model, temp_db_path):
    """Test adding documents successfully."""
    with patch('src.config.config', {"vector_db_path": temp_db_path, "embedding_batch_size": 32}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_success")
        docs = ["Test document 1", "Test document 2"]
        retriever.add_documents(docs)
        assert retriever.collection.count() == 2

def test_retrieve_empty_query(mock_model, temp_db_path):
    """Test retrieval with empty query."""
    with patch('config.config', {"vector_db_path": temp_db_path, "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        result = retriever.retrieve_top_documents("")
        assert result == []

def test_retrieve_no_documents(mock_model, temp_db_path):
    """Test retrieval when no documents are loaded."""
    with patch('config.config', {"vector_db_path": temp_db_path, "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        result = retriever.retrieve_top_documents("test query")
        assert result == []

def test_retrieve_success(mock_model, temp_db_path):
    """Test successful retrieval."""
    with patch('src.config.config', {"vector_db_path": temp_db_path, "top_k": 3, "similarity_threshold": 0.5}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_retrieve")
        docs = ["Relevant document"]
        retriever.add_documents(docs)
        # Mock the query method properly
        mock_result = {'documents': [['Relevant document']], 'distances': [[0.3]]}
        with patch.object(retriever.collection, 'query', return_value=mock_result):
            result = retriever.retrieve_top_documents("test query")
            assert len(result) == 1
            assert result[0] == "Relevant document"

def test_cache_persistence(mock_model, temp_db_path):
    """Test that cache is persisted to disk."""
    with patch('src.config.config', {"vector_db_path": temp_db_path, "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_cache")
        # Simulate adding to cache
        retriever.query_cache[("test", 3)] = ["cached doc"]
        retriever._save_cache()
        cache_file = os.path.join(temp_db_path, "query_cache.pkl")
        assert os.path.exists(cache_file)
        # Load new instance and check cache
        retriever2 = DocumentRetriever(mock_model)
        retriever2.collection = retriever2.client.get_or_create_collection("test_cache2")  # Different collection
        assert ("test", 3) in retriever2.query_cache