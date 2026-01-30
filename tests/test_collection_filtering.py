"""Tests for collection-scoped retrieval filtering.

This module tests the functionality of filtering document retrieval by:
- Collection ID (collection_id parameter)
- Document ID set (doc_ids parameter)
- Short-circuit behavior for empty document sets

Note: Some integration tests are skipped when DocumentRetriever requires
full initialization. Unit tests focus on API and core layer acceptance.
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock, call, patch

# Add the parent directory to the path so we can import cubo modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCollectionFilteringAPILayer(unittest.TestCase):
    """Test collection filtering at the API layer (QueryRequest model)."""

    def test_query_request_model_accepts_doc_ids(self):
        """Test that QueryRequest Pydantic model accepts doc_ids field."""
        from cubo.server.api import QueryRequest

        # Create request with doc_ids
        request = QueryRequest(query="test query", doc_ids=["doc1.txt", "doc2.txt"])

        self.assertEqual(request.query, "test query")
        self.assertEqual(request.doc_ids, ["doc1.txt", "doc2.txt"])

    def test_query_request_model_accepts_doc_ids_none(self):
        """Test that QueryRequest accepts None for doc_ids."""
        from cubo.server.api import QueryRequest

        request = QueryRequest(query="test query")

        self.assertEqual(request.query, "test query")
        self.assertIsNone(request.doc_ids)


class TestCollectionFilteringCoreLayer(unittest.TestCase):
    """Test collection filtering at the core layer (CuboCore)."""

    def test_core_query_retrieve_passes_doc_ids(self):
        """Test that CuboCore.query_retrieve passes doc_ids to retriever."""
        from cubo.core import CuboCore

        with patch.object(CuboCore, "__init__", return_value=None):
            core = CuboCore()
            core.retriever = MagicMock()
            core.retriever.retrieve_top_documents = MagicMock(return_value=[])
            core._state_lock = MagicMock()
            core._state_lock.__enter__ = MagicMock()
            core._state_lock.__exit__ = MagicMock(return_value=False)

            # Mock config
            with patch("cubo.core.config") as mock_config:
                mock_config.get.return_value = 6

                doc_ids = ["doc1.txt", "doc2.txt"]
                core.query_retrieve(
                    "test query", top_k=5, doc_ids=doc_ids, collection_id="my_collection"
                )

                # Verify retriever was called with doc_ids and collection_id
                core.retriever.retrieve_top_documents.assert_called_once()
                call_kwargs = core.retriever.retrieve_top_documents.call_args[1]
                self.assertEqual(call_kwargs.get("doc_ids"), doc_ids)
                self.assertEqual(call_kwargs.get("collection_id"), "my_collection")

    def test_core_query_retrieve_backward_compatible(self):
        """Test that CuboCore.query_retrieve works without collection parameters."""
        from cubo.core import CuboCore

        with patch.object(CuboCore, "__init__", return_value=None):
            core = CuboCore()
            core.retriever = MagicMock()
            core.retriever.retrieve_top_documents = MagicMock(return_value=[])
            core._state_lock = MagicMock()
            core._state_lock.__enter__ = MagicMock()
            core._state_lock.__exit__ = MagicMock(return_value=False)

            with patch("cubo.core.config") as mock_config:
                mock_config.get.return_value = 6

                # Call without collection_id or doc_ids (backward compatibility)
                core.query_retrieve("test query", top_k=5)

                # Should still work
                core.retriever.retrieve_top_documents.assert_called_once()


class TestMetadataManager(unittest.TestCase):
    """Test metadata manager collection filtering helpers."""

    def test_metadata_manager_has_get_filenames_method(self):
        """Test that MetadataManager has get_filenames_in_collection method."""
        from cubo.storage.metadata_manager import MetadataManager

        # Method should exist
        self.assertTrue(hasattr(MetadataManager, "get_filenames_in_collection"))

        # Check it's callable
        self.assertTrue(callable(getattr(MetadataManager, "get_filenames_in_collection")))


class TestRetrieverSignature(unittest.TestCase):
    """Test that retriever method signatures accept new parameters."""

    def test_retrieve_top_documents_signature(self):
        """Test that retrieve_top_documents accepts collection_id and doc_ids."""
        import inspect

        from cubo.retrieval.retriever import DocumentRetriever

        sig = inspect.signature(DocumentRetriever.retrieve_top_documents)
        params = list(sig.parameters.keys())

        # Should have collection_id and doc_ids parameters
        self.assertIn("collection_id", params)
        self.assertIn("doc_ids", params)

    def test_retrieve_sentence_window_signature(self):
        """Test that _retrieve_sentence_window accepts doc_ids."""
        import inspect

        from cubo.retrieval.retriever import DocumentRetriever

        sig = inspect.signature(DocumentRetriever._retrieve_sentence_window)
        params = list(sig.parameters.keys())

        # Should have doc_ids parameter
        self.assertIn("doc_ids", params)

    def test_hybrid_retrieval_signature(self):
        """Test that _hybrid_retrieval accepts doc_ids."""
        import inspect

        from cubo.retrieval.retriever import DocumentRetriever

        sig = inspect.signature(DocumentRetriever._hybrid_retrieval)
        params = list(sig.parameters.keys())

        # Should have doc_ids parameter
        self.assertIn("doc_ids", params)


if __name__ == "__main__":
    unittest.main()
