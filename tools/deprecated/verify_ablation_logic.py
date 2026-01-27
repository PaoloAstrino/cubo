#!/usr/bin/env python3
"""
Ablation Verification Script.
Checks if dense_weight=0.0 actually skips embedding computation.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np

from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever


def verify_dense_skip():
    print("Verifying if dense_weight=0.0 skips embedding computation...")

    # Mock components
    mock_model = MagicMock()
    # Mock dimensions
    mock_model.get_sentence_embedding_dimension.return_value = 384
    # Mock encoding - we want to see if this is called
    mock_model.encode.return_value = np.zeros((1, 384))

    # Initialize retriever with mocked model
    retriever = DocumentRetriever(model=mock_model)

    # Mock executor components
    retriever.executor.inference_threading = MagicMock()
    retriever.executor.inference_threading.generate_embeddings_threaded.return_value = [
        np.zeros(384)
    ]

    # Mock collection query to return empty
    retriever.collection.query = MagicMock(
        return_value={"documents": [], "metadatas": [], "distances": []}
    )

    # Mock BM25 search to return empty (so we don't crash on fusion)
    retriever.bm25.search = MagicMock(return_value=[])

    # CASE 1: Normal Hybrid (dense_weight > 0)
    print("\n--- Case 1: dense_weight = 0.7 ---")
    strategy = {"dense_weight": 0.7, "bm25_weight": 0.3}
    retriever._retrieve_sentence_window("test query", top_k=5, strategy=strategy)

    # Check if embedding was generated
    call_count_1 = retriever.executor.inference_threading.generate_embeddings_threaded.call_count
    print(f"Embedding calls: {call_count_1}")
    if call_count_1 > 0:
        print("✓ PASS: Embeddings computed for hybrid search")
    else:
        print("✗ FAIL: Embeddings NOT computed for hybrid search")

    # CASE 2: Sparse Only (dense_weight = 0.0)
    print("\n--- Case 2: dense_weight = 0.0 ---")
    # Reset mock
    retriever.executor.inference_threading.generate_embeddings_threaded.reset_mock()

    strategy = {"dense_weight": 0.0, "bm25_weight": 1.0}
    retriever._retrieve_sentence_window("test query", top_k=5, strategy=strategy)

    # Check if embedding was generated
    call_count_2 = retriever.executor.inference_threading.generate_embeddings_threaded.call_count
    print(f"Embedding calls: {call_count_2}")

    if call_count_2 == 0:
        print("✓ PASS: Embeddings SKIPPED for dense_weight=0.0")
    else:
        print("✗ FAIL: Embeddings COMPUTED despite dense_weight=0.0")
        print("  -> This confirms the 'Dense (BM25 only)' label confusion/latency bug.")


if __name__ == "__main__":
    # Ensure config allows mocking
    config.set("vector_store_backend", "memory")
    verify_dense_skip()
