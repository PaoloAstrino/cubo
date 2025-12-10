"""
Debug script to test retrieval behavior across multiple documents.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from cubo.core import CuboCore
from cubo.security.security import security_manager


def test_retrieval():
    """Test retrieval with multiple queries."""
    print("=" * 80)
    print("RETRIEVAL DEBUG TEST")
    print("=" * 80)

    # Initialize Cubo core and retriever
    print("\n1. Loading CuboCore components...")
    core = CuboCore()
    core.initialize_components()
    retriever = core.retriever
    print(f"   Model loaded: {core.model is not None}")
    print("   Retriever initialized")
    print(f"   Current documents in session: {retriever.current_documents}")

    # Check what's in the database
    print("\n2. Checking database contents...")
    try:
        collection_info = retriever.debug_collection_info()
        print(f"   Total chunks in DB: {collection_info.get('total_chunks', 0)}")
        print(f"   Current session docs: {collection_info.get('current_session_docs', 0)}")
        print(f"   Filenames in session: {collection_info.get('current_session_filenames', [])}")
    except:
        print(f"   Current session docs: {len(retriever.current_documents)}")

    # Get all metadata to see what files are actually in the database
    print("\n3. Querying all documents in vector store...")
    all_data = retriever.collection.get()
    if all_data and all_data.get("metadatas"):
        filenames_in_db = set()
        for metadata in all_data["metadatas"]:
            if "filename" in metadata:
                filenames_in_db.add(metadata["filename"])
        print(f"   Files in database: {sorted(filenames_in_db)}")
        print(f"   Total chunks: {len(all_data['ids'])}")

        # Count chunks per file
        from collections import Counter

        file_counts = Counter(m.get("filename", "Unknown") for m in all_data["metadatas"])
        print("\n   Chunks per file:")
        for filename, count in file_counts.most_common():
            print(f"     - {filename}: {count} chunks")
    else:
        print("   No documents found in database!")
        return

    # Check auto-merging retriever
    print("\n4. Checking auto-merging retriever...")
    if retriever.auto_merging_retriever:
        auto_collection = retriever.auto_merging_retriever.collection.get()
        if auto_collection and auto_collection.get("metadatas"):
            auto_filenames = set()
            for metadata in auto_collection["metadatas"]:
                if "filename" in metadata:
                    auto_filenames.add(metadata["filename"])
            print(f"   Files in auto-merging: {sorted(auto_filenames)}")
            print(f"   Total auto-merging chunks: {len(auto_collection['ids'])}")
        else:
            print("   No documents in auto-merging collection")
    else:
        print("   Auto-merging retriever not available")

    # Test queries
    test_queries = [
        "tell me about the frog",
        "tell me about the horse",
        "tell me about the lion",
        "what is a cat",
        "describe the elephant",
    ]

    print("\n" + "=" * 80)
    print("TESTING QUERIES")
    print("=" * 80)

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{security_manager.scrub(query)}'")
        print(f"{'='*80}")

        # Test sentence window retrieval
        print("\n  [Sentence Window Retrieval]")
        try:
            sentence_results = retriever._retrieve_sentence_window(query, top_k=3)
            print(f"  Retrieved {len(sentence_results)} results")
            for i, result in enumerate(sentence_results, 1):
                filename = result.get("metadata", {}).get("filename", "Unknown")
                similarity = result.get("similarity", 0)
                doc_preview = result.get("document", "")[:100]
                print(f"    {i}. {filename} (similarity: {similarity:.4f})")
                print(f"       Preview: {doc_preview}...")
        except Exception as e:
            print(f"  Error: {e}")

        # Test auto-merging retrieval
        print("\n  [Auto-Merging Retrieval]")
        try:
            auto_results = retriever._retrieve_auto_merging_safe(query, top_k=3)
            print(f"  Retrieved {len(auto_results)} results")
            for i, result in enumerate(auto_results, 1):
                filename = result.get("metadata", {}).get("filename", "Unknown")
                similarity = result.get("similarity", 0)
                doc_preview = result.get("document", "")[:100]
                print(f"    {i}. {filename} (similarity: {similarity:.4f})")
                print(f"       Preview: {doc_preview}...")
        except Exception as e:
            print(f"  Error: {e}")

        # Test hybrid retrieval
        print("\n  [Hybrid Retrieval (Combined)]")
        try:
            hybrid_results = retriever.retrieve_top_documents(query, top_k=6)
            print(f"  Retrieved {len(hybrid_results)} results")
            for i, result in enumerate(hybrid_results, 1):
                filename = result.get("metadata", {}).get("filename", "Unknown")
                similarity = result.get("similarity", 0)
                doc_preview = result.get("document", "")[:100]
                print(f"    {i}. {filename} (similarity: {similarity:.4f})")
                print(f"       Preview: {doc_preview}...")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_retrieval()
