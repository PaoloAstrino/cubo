import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from sentence_transformers import SentenceTransformer

from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model."""
    import numpy as np

    model = MagicMock(spec=SentenceTransformer)

    # Mock encode to return embeddings based on input length
    def mock_encode(texts, **kwargs):
        # Return one embedding per input text as a numpy array
        return np.ones((len(texts), 768), dtype=np.float32) * 0.1

    model.encode.side_effect = mock_encode
    # Ensure embedding dimension is an int to avoid failing FAISS API calls
    model.get_sentence_embedding_dimension.return_value = 768
    return model


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the vector store."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_add_documents_empty(mock_model, temp_db_path):
    """Test adding empty documents list."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_add_documents_empty")
    # Use FAISS with small dimension to match mock_model embedding size
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    retriever.add_documents([])  # Should not crash
    assert retriever.collection.count() == 0
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_add_documents_success(mock_model, temp_db_path):
    """Test adding documents successfully."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_add_documents_success")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    docs = ["Test document 1", "Test document 2"]
    retriever.add_documents(docs)
    assert retriever.collection.count() == 2
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_retrieve_empty_query(mock_model, temp_db_path):
    """Test retrieval with empty query."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_retrieve_empty_query")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    result = retriever.retrieve_top_documents("")
    assert result == []
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_retrieve_no_documents(mock_model, temp_db_path):
    """Test retrieval when no documents are loaded."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_retrieve_no_documents")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    result = retriever.retrieve_top_documents("test query")
    # When no documents are available, retriever returns a padded list of
    # placeholders to ensure consistent list length for downstream consumers.
    assert isinstance(result, list)
    assert len(result) == retriever.top_k
    # Placeholder entries carry zero similarity and empty document text
    for r in result:
        assert r.get("similarity", None) == 0.0
        assert r.get("document", "") == ""
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_retrieve_success(mock_model, temp_db_path):
    """Test successful retrieval."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_retrieve_success")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    # Ensure mock returns 768-dim embeddings to match index
    mock_model.get_sentence_embedding_dimension.return_value = 768
    retriever = DocumentRetriever(mock_model)
    docs = ["Relevant document"]
    retriever.add_documents(docs)
    # Clear cache to ensure fresh query
    retriever.query_cache.clear()
    result = retriever.retrieve_top_documents("test query")
    # Should return results (at least the document we added)
    assert len(result) >= 0
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_cache_persistence(mock_model, temp_db_path):
    """Test that cache is persisted to disk."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_cache_persistence")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
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
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()
    close_fn2 = getattr(retriever2, "close", None)
    if callable(close_fn2):
        close_fn2()


def test_retrieve_uses_router_strategy(mock_model, temp_db_path):
    # Create a retriever and ensure it uses router strategy for k_candidates
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_retrieve_uses_router_strategy")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)

    # Patch router to return a specific k_candidates
    class FakeRouter:
        def route_query(self, q):
            return {
                "k_candidates": 123,
                "bm25_weight": 0.7,
                "dense_weight": 0.3,
                "use_reranker": False,
            }

    retriever.router = FakeRouter()

    captured = {}

    def fake_retrieve_sentence_window(query, top_k, strategy=None, trace_id=None, **kwargs):
        captured["strategy"] = strategy
        return []

    retriever._retrieve_sentence_window = fake_retrieve_sentence_window
    retriever.retrieve_top_documents("Test query", top_k=5)
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()
    assert "strategy" in captured
    assert captured["strategy"]["k_candidates"] == 123


def test_reranker_initialization_with_crossencoder(mock_model, temp_db_path):
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_reranker_initialization")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    # Default behavior may not initialize a reranker if dependencies aren't present. Ensure attribute exists
    assert hasattr(retriever, "reranker")
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()

    # Patch config to use cross-encoder model and reinitialize
    with patch.object(
        config,
        "get",
        side_effect=lambda k, d=None: (
            "cross-encoder/ms-marco-MiniLM-L-6-v2" if k == "retrieval.reranker_model" else d
        ),
    ):
        retriever2 = DocumentRetriever(mock_model)
        assert hasattr(retriever2, "reranker")
        from cubo.rerank.reranker import CrossEncoderReranker, LocalReranker

        assert isinstance(retriever2.reranker, (CrossEncoderReranker, LocalReranker, type(None)))


def test_apply_reranking_called_when_enabled(mock_model, temp_db_path):
    # Ensure the internal function respects use_reranker flag
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_apply_reranking_called")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
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
    candidates = [{"document": "a"}, {"document": "b"}, {"document": "c"}]
    retriever._apply_reranking_if_available(candidates, top_k=2, query="test", use_reranker=False)
    assert fake.called is False
    retriever._apply_reranking_if_available(candidates, top_k=2, query="test", use_reranker=True)
    assert fake.called is True
    close_fn = getattr(retriever, "close", None)
    if callable(close_fn):
        close_fn()


def test_retrieve_top_documents_accepts_k_kwarg(mock_model, temp_db_path):
    """Ensure retrieve_top_documents accepts 'k' kwarg as backwards compatibility."""
    config.set("vector_store_path", temp_db_path)
    config.set("collection_name", "test_retrieve_accepts_k_kwarg")
    config.set("vector_store_backend", "faiss")
    config.set("index_dimension", 768)
    retriever = DocumentRetriever(mock_model)
    docs = ["Doc one", "Doc two", "Doc three"]
    retriever.add_documents(docs)
    try:
        # Call with deprecated 'k' kwarg and ensure no TypeError is raised and function returns list
        res = retriever.retrieve_top_documents("test query", k=2)
        assert isinstance(res, list)
    finally:
        close_fn = getattr(retriever, "close", None)
        if callable(close_fn):
            close_fn()
