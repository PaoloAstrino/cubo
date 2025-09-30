"""
Test script for custom auto-merging retrieval implementation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
from src.custom_auto_merging import AutoMergingRetriever
from src.logger import logger

def test_custom_auto_merging():
    """Test the custom auto-merging retrieval system."""

    # Initialize model
    model_path = "./models/embeddinggemma-300m"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return False

    try:
        model = SentenceTransformer(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # Initialize auto-merging retriever
    try:
        retriever = AutoMergingRetriever(model)
        logger.info("Auto-merging retriever initialized")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return False

    # Test document addition
    test_file = "./data/horse_story.txt"
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return False

    try:
        success = retriever.add_document(test_file)
        if success:
            logger.info("Document added successfully")
        else:
            logger.error("Failed to add document")
            return False
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        return False

    # Test retrieval
    test_queries = [
        "What is the story about?",
        "Who are the main characters?",
        "What happens in the end?"
    ]

    for query in test_queries:
        try:
            results = retriever.retrieve(query, top_k=3)
            logger.info(f"Query: '{query}' - Retrieved {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: similarity={result.get('similarity', 0):.3f}")
                logger.info(f"    Text preview: {result.get('document', '')[:100]}...")
        except Exception as e:
            logger.error(f"Error retrieving for query '{query}': {e}")
            return False

    # Test loaded documents
    loaded_docs = retriever.get_loaded_documents()
    logger.info(f"Loaded documents: {loaded_docs}")

    logger.info("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_custom_auto_merging()
    sys.exit(0 if success else 1)