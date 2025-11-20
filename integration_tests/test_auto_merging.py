#!/usr/bin/env python3
"""
Test script for auto-merging retrieval with local embedding model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cubo.deduplication.custom_auto_merging import AutoMergingRetriever
from llama_index.core import Document
from src.cubo.config import config
import numpy as np


class DummyModel:
    def encode(self, texts):
        # Return 2-dimensional vectors for all texts
        return np.array([[0.1, 0.1] for _ in texts], dtype='float32')

def test_auto_merging():
    """Test auto-merging retrieval with local model."""
    print("🧪 Testing Auto-Merging Retrieval with Local Embedding Model")
    print("=" * 60)

    # Create test documents
    test_docs = [
        Document(text="The frog lives in a wetland near a pond. He helps other animals and is very friendly."),
        Document(text="Thunder is a horse who lives on a farm. He enjoys watching sunsets and helping the farmer."),
        Document(text="The frog can jump very high and swim well. He has many friends in the wetland."),
        Document(text="Thunder runs fast and is strong. He likes to graze on grass in the pasture.")
    ]

    print(f"📄 Created {len(test_docs)} test documents")

    # Initialize retriever
    print("🔧 Initializing auto-merging retriever...")
    # Use a dummy model for quick integration test runs to avoid heavy model downloads
    config.set('auto_merge_index_dimension', 2)
    retriever = AutoMergingRetriever(DummyModel())

    # Build index by creating temporary files for each document and adding them
    print("🏗️ Building index via file ingestion...")
    import tempfile
    from pathlib import Path
    tmpdir = Path(tempfile.mkdtemp())
    for i, doc in enumerate(test_docs, 1):
        fp = tmpdir / f"doc_{i}.txt"
        with open(fp, 'w', encoding='utf-8') as fh:
            fh.write(doc.text)
        added = retriever.add_document(str(fp))
        assert added is True

    # Test queries
    test_queries = [
        "What animals are mentioned?",
        "Where does the frog live?",
        "What does Thunder like to do?",
        "Compare the frog and horse's abilities"
    ]

    print("\n🔍 Testing retrieval...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = retriever.retrieve(query, top_k=3)
            print(f"Retrieved {len(results)} results:")
            for i, result in enumerate(results, 1):
                content = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
                score = result.get('score', 0)
                print(f"  {i}. {content} (score: {score:.3f})")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n✅ Auto-merging retrieval test completed!")

if __name__ == "__main__":
    test_auto_merging()