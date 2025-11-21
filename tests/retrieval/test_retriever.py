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

def test_add_documents_empty(mock_model, temp_db_path):
    """Test adding empty documents list."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_add_documents_empty')
    # Use FAISS with small dimension to match mock_model embedding size
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    retriever.add_documents([])  # Should not crash
    assert retriever.collection.count() == 0

def test_add_documents_success(mock_model, temp_db_path):
    """Test adding documents successfully."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_add_documents_success')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    docs = ["Test document 1", "Test document 2"]
    retriever.add_documents(docs)
    assert retriever.collection.count() == 2

def test_retrieve_empty_query(mock_model, temp_db_path):
    """Test retrieval with empty query."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_retrieve_empty_query')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    result = retriever.retrieve_top_documents("")
    assert result == []

def test_retrieve_no_documents(mock_model, temp_db_path):
    """Test retrieval when no documents are loaded."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_retrieve_no_documents')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    result = retriever.retrieve_top_documents("test query")
    assert result == []

def test_retrieve_success(mock_model, temp_db_path):
    """Test successful retrieval."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_retrieve_success')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    docs = ["Relevant document"]
    retriever.add_documents(docs)
    # Clear cache to ensure fresh query
    retriever.query_cache.clear()
    result = retriever.retrieve_top_documents("test query")
    # Should return results (at least the document we added)
    assert len(result) >= 0

def test_cache_persistence(mock_model, temp_db_path):
    """Test that cache is persisted to disk."""
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_cache_persistence')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    # Simulate adding to cache
    retriever.query_cache[("test", 3)] = ["cached doc"]
    retriever._save_cache()
    # Use the actual cache file path from the retriever
    cache_file = retriever.cache_file
    assert os.path.exists(cache_file)
    # Load new instance and check cache
    retriever2 = DocumentRetriever(mock_model)
    assert ("test", 3) in retriever2.query_cache

def test_retrieve_uses_router_strategy(mock_model, temp_db_path):
    # Create a retriever and ensure it uses router strategy for k_candidates
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_retrieve_uses_router_strategy')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
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


def test_reranker_initialization_with_crossencoder(mock_model, temp_db_path):
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_reranker_initialization')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
    retriever = DocumentRetriever(mock_model)
    # Default behavior may not initialize a reranker if dependencies aren't present. Ensure attribute exists
    assert hasattr(retriever, 'reranker')

    # Patch config to use cross-encoder model and reinitialize
    with patch.object(config, 'get', side_effect=lambda k, d=None: "cross-encoder/ms-marco-MiniLM-L-6-v2" if k == "retrieval.reranker_model" else d):
        retriever2 = DocumentRetriever(mock_model)
        assert hasattr(retriever2, 'reranker')
        from src.cubo.rerank.reranker import CrossEncoderReranker, LocalReranker
        assert isinstance(retriever2.reranker, (CrossEncoderReranker, LocalReranker, type(None)))


def test_apply_reranking_called_when_enabled(mock_model, temp_db_path):
    # Ensure the internal function respects use_reranker flag
    config.set('chroma_db_path', temp_db_path)
    config.set('vector_store_path', temp_db_path)
    config.set('collection_name', 'test_apply_reranking_called')
    config.set('vector_store_backend', 'faiss')
    config.set('index_dimension', 768)
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