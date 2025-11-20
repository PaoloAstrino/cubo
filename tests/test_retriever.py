import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer
from src.cubo.retrieval.retriever import DocumentRetriever
from src.cubo.config import config

try:
    import chromadb.config
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model."""
    model = MagicMock(spec=SentenceTransformer)
    # Mock encode to return embeddings based on input length
    def mock_encode(texts, convert_to_tensor=True):
        mock_tensor = MagicMock()
        # Return one embedding per input text
        embeddings = [[0.1] * 768 for _ in texts]
        mock_tensor.tolist.return_value = embeddings
        return mock_tensor
    model.encode.side_effect = mock_encode
    return model

@pytest.fixture
def temp_db_path():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_add_documents_empty(mock_model, temp_db_path):
    """Test adding empty documents list."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "embedding_batch_size": 32}):
        # Use a unique collection name to avoid conflicts
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_empty")
        retriever.add_documents([])  # Should not crash
        assert retriever.collection.count() == 0

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_add_documents_success(mock_model, temp_db_path):
    """Test adding documents successfully."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "embedding_batch_size": 32}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_success")
        docs = ["Test document 1", "Test document 2"]
        retriever.add_documents(docs)
        assert retriever.collection.count() == 2

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_retrieve_empty_query(mock_model, temp_db_path):
    """Test retrieval with empty query."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_empty_query")
        result = retriever.retrieve_top_documents("")
        assert result == []

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_retrieve_no_documents(mock_model, temp_db_path):
    """Test retrieval when no documents are loaded."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_no_docs")
        result = retriever.retrieve_top_documents("test query")
        assert result == []

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_retrieve_success(mock_model, temp_db_path):
    """Test successful retrieval."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "top_k": 3, "similarity_threshold": 0.5}):
        # Mock the collection query method before creating the retriever
        mock_result = {'documents': [['Relevant document']], 'distances': [[0.3]], 'metadatas': [[{'filename': 'test_doc_0.txt'}]]}
        with patch('chromadb.api.models.Collection.Collection.query', return_value=mock_result) as mock_query:
            retriever = DocumentRetriever(mock_model)
            retriever.collection = retriever.client.get_or_create_collection("test_retrieve")
            docs = ["Relevant document"]
            retriever.add_documents(docs)
            # Clear cache to ensure query is called
            retriever.query_cache.clear()
            result = retriever.retrieve_top_documents("test query")
            mock_query.assert_called_once()
            assert len(result) == 1
            assert result[0]['document'] == "Relevant document"

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_cache_persistence(mock_model, temp_db_path):
    """Test that cache is persisted to disk."""
    with patch('src.cubo.config.config', {"chroma_db_path": temp_db_path, "vector_store_backend": "chroma", "top_k": 3}):
        retriever = DocumentRetriever(mock_model)
        retriever.collection = retriever.client.get_or_create_collection("test_cache")
        # Simulate adding to cache
        retriever.query_cache[("test", 3)] = ["cached doc"]
        retriever._save_cache()
        # Use the actual cache file path from the retriever
        cache_file = retriever.cache_file
        assert os.path.exists(cache_file)
        # Load new instance and check cache (within same config context)
        retriever2 = DocumentRetriever(mock_model)
        retriever2.collection = retriever2.client.get_or_create_collection("test_cache2")  # Different collection
        assert ("test", 3) in retriever2.query_cache

@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not available")
def test_retrieve_uses_router_strategy(mock_model):
    # Create a retriever and ensure it uses router strategy for k_candidates
    retriever = DocumentRetriever(mock_model)
    # Patch router to return a specific k_candidates
    class FakeRouter:
        def route_query(self, q):
            return {'k_candidates': 123, 'bm25_weight': 0.7, 'dense_weight': 0.3, 'use_reranker': False}
    retriever.router = FakeRouter()

    captured = {}
    def fake_retrieve_sentence_window(q, top_k, strategy=None):
        captured['strategy'] = strategy
        return []

    retriever._retrieve_sentence_window = fake_retrieve_sentence_window
    res = retriever.retrieve_top_documents('Test query', top_k=5)
    assert 'strategy' in captured
    assert captured['strategy']['k_candidates'] == 123


def test_reranker_initialization_with_crossencoder(mock_model):
    retriever = DocumentRetriever(mock_model)
    # Default behavior may not initialize a reranker if dependencies aren't present. Ensure attribute exists
    assert hasattr(retriever, 'reranker')

    # Patch config to use cross-encoder model and reinitialize
    with patch('src.cubo.config.config', {"retrieval.reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}):
        retriever2 = DocumentRetriever(mock_model)
        assert hasattr(retriever2, 'reranker')
        from src.cubo.rerank.reranker import CrossEncoderReranker, LocalReranker
        assert isinstance(retriever2.reranker, (CrossEncoderReranker, LocalReranker, type(None)))


    def test_apply_reranking_called_when_enabled(mock_model):
        # Ensure the internal function respects use_reranker flag
        retriever = DocumentRetriever(mock_model)
        class FakeReranker:
            def __init__(self):
                self.called = False
            def rerank(self, query, candidates, max_results=None):
                self.called = True
                # return the same candidates to keep semantics
                return candidates

        fake = FakeReranker()
        retriever.reranker = fake
        candidates = [{'document': 'a'}, {'document': 'b'}, {'document': 'c'}]
        out = retriever._apply_reranking_if_available(candidates, top_k=2, query='test', use_reranker=False)
        assert fake.called is False
        out = retriever._apply_reranking_if_available(candidates, top_k=2, query='test', use_reranker=True)
        assert fake.called is True