"""
Simplified Full RAG Pipeline E2E Test

Tests core retrieval functionality with minimal complexity.
Uses existing test data files instead of creating temporary ones.
"""

import pytest

pytest.importorskip("torch")
from pathlib import Path

from cubo.embeddings.lazy_model_manager import get_lazy_model_manager
from cubo.ingestion.document_loader import DocumentLoader
from cubo.retrieval.retriever import DocumentRetriever


class TestSimplifiedRAGPipeline:
    """Simplified E2E tests using actual data files."""

    def test_load_and_query_real_files(self):
        """
        E2E: Load real test files → Query → Verify results.

        This is the simplest possible E2E test that validates the core pipeline.
        """
        # Find test data files
        data_dir = Path("data")
        if not data_dir.exists():
            pytest.skip("data/ directory not found")

        # Find any .txt files to test with
        txt_files = list(data_dir.glob("*.txt"))
        if len(txt_files) < 2:
            pytest.skip("Need at least 2 .txt files in data/ for testing")

        # Use first 2 files
        test_files = txt_files[:2]

        # 1. Load documents
        loader = DocumentLoader()
        model = get_lazy_model_manager().get_model()
        retriever = DocumentRetriever(model=model, use_sentence_window=True, top_k=5)

        total_chunks = 0
        for file_path in test_files:
            chunks = loader.load_single_document(str(file_path))
            if chunks:
                retriever.add_document(str(file_path), chunks)
                total_chunks += len(chunks)
                print(f"✓ Loaded {len(chunks)} chunks from {file_path.name}")

        assert total_chunks > 0, "No chunks loaded"
        print(f"✓ Total: {total_chunks} chunks from {len(test_files)} files")

        # 2. Query
        # Use a generic query that should match any story content
        query = "What happens in this story?"
        results = retriever.retrieve_top_documents(query, top_k=3)

        assert len(results) > 0, "No results returned"
        print(f"✓ Retrieved {len(results)} results")

        # 3. Verify results structure
        for i, result in enumerate(results):
            assert "document" in result, f"Result {i} missing 'document' key"
            assert len(result["document"]) > 0, f"Result {i} has empty document"
            assert "metadata" in result, f"Result {i} missing 'metadata'"
            assert "similarity" in result, f"Result {i} missing 'similarity'"

        # 4. Verify sources
        filenames = [r.get("metadata", {}).get("filename", "") for r in results]
        assert any(filenames), "No filenames in results"

        print(f"✓ Results include sources: {set(filenames)}")
        print(f"✓ Top result: {results[0]['document'][:100]}...")

        print("\n✅ SIMPLIFIED RAG PIPELINE TEST PASSED")

    def test_multifile_retrieval(self):
        """
        Test that queries can retrieve from multiple files.
        """
        data_dir = Path("data")
        if not data_dir.exists():
            pytest.skip("data/ directory not found")

        txt_files = list(data_dir.glob("*.txt"))[:3]  # Use up to 3 files
        if len(txt_files) < 2:
            pytest.skip("Need at least 2 files")

        loader = DocumentLoader()
        model = get_lazy_model_manager().get_model()
        retriever = DocumentRetriever(model=model)

        for file_path in txt_files:
            chunks = loader.load_single_document(str(file_path))
            if chunks:
                retriever.add_document(str(file_path), chunks)

        # Query
        results = retriever.retrieve_top_documents("story", top_k=10)

        # Get unique source files
        sources = set(r.get("metadata", {}).get("filename", "") for r in results)

        # Should have results from multiple files
        assert len(sources) >= 2, f"Results only from {len(sources)} file(s), expected >=2"

        print(f"✓ Retrieved from {len(sources)} different files")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
