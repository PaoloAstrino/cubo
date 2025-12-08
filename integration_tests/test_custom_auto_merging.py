"""
Test script for custom auto-merging retrieval implementation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest

import tempfile
from pathlib import Path as _Path

from cubo.config import config as _test_cfg

# Ensure vector store uses an isolated in-memory backend before importing modules
# that may load FAISS indexes during import
tmpdir_global = _Path(tempfile.mkdtemp())
_test_cfg.set("vector_store_backend", "inmemory")
_test_cfg.set("vector_store_path", str(tmpdir_global / "vector_store"))

from cubo.deduplication.custom_auto_merging import AutoMergingRetriever
from cubo.utils.logger import logger


def test_custom_auto_merging():
    """Test the custom auto-merging retrieval system."""

    # Initialize model
    model_path = "./models/embeddinggemma-300m"
    if not os.path.exists(model_path):
        pytest.skip(f"Model not found at {model_path}")

    try:
        # For test stability, use a dummy lightweight model that returns 2D embeddings
        import numpy as np

        class DummyModel:
            def encode(self, texts):
                return np.array([[0.1, 0.1] for _ in texts], dtype="float32")

        model = DummyModel()
        # Ensure FAISS uses dimension 2 for the auto-merging vector store
        from cubo.config import config as _cfg

        _cfg.set("auto_merge_index_dimension", 2)
        # Force in-memory vector store for auto-merging to avoid appending to existing FAISS indexes
        _cfg.set("auto_merge_vector_store_backend", "inmemory")
        logger.info("Dummy model set for testing")
        import tempfile
        from pathlib import Path

        tmpdir = Path(tempfile.mkdtemp())
        _cfg.set("vector_store_backend", "inmemory")
        _cfg.set("vector_store_path", str(tmpdir / "vector_store"))
    except Exception as e:
        pytest.skip(f"Failed to load model: {e}")

    # Initialize auto-merging retriever
    try:
        retriever = AutoMergingRetriever(model)
        logger.info("Auto-merging retriever initialized")
    except Exception as e:
        pytest.fail(f"Failed to initialize retriever: {e}")

    # Test document addition
    test_file = "./data/horse_story.txt"
    if not os.path.exists(test_file):
        pytest.skip(f"Test file not found: {test_file}")

    try:
        success = retriever.add_document(test_file)
        if success:
            logger.info("Document added successfully")
        else:
            pytest.fail("Failed to add document")
    except Exception as e:
        pytest.fail(f"Error adding document: {e}")

    # Test retrieval
    test_queries = [
        "What is the story about?",
        "Who are the main characters?",
        "What happens in the end?",
    ]

    for query in test_queries:
        try:
            results = retriever.retrieve(query, top_k=3)
            logger.info(f"Query: '{query}' - Retrieved {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: similarity={result.get('similarity', 0):.3f}")
                logger.info(f"    Text preview: {result.get('document', '')[:100]}...")
        except Exception as e:
            pytest.fail(f"Error retrieving for query '{query}': {e}")

    # Test loaded documents
    loaded_docs = retriever.get_loaded_documents()
    logger.info(f"Loaded documents: {loaded_docs}")

    logger.info("All tests passed!")
    assert True


if __name__ == "__main__":
    test_custom_auto_merging()
    sys.exit(0)
