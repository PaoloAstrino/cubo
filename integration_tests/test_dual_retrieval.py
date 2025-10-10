#!/usr/bin/env python3
"""
Simple test script for dual retrieval system (sentence window + auto-merging)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
from retriever import DocumentRetriever
from document_loader import DocumentLoader
from logger import logger

def test_dual_retrieval():
    """Test the dual retrieval system with both methods."""

    print("🔄 Testing Dual Retrieval System")
    print("=" * 40)

    # Load model
    model_path = "./models/embeddinggemma-300m"
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path)
    print("✅ Model loaded")

    # Initialize retriever with both methods enabled
    print("Initializing dual retriever...")
    retriever = DocumentRetriever(
        model=model,
        use_sentence_window=True,
        use_auto_merging=True,
        auto_merge_for_complex=True
    )
    print("✅ Dual retriever initialized")

    # Load documents
    document_loader = DocumentLoader()
    data_dir = Path("./data")

    story_files = ["frog_story.txt", "horse_story.txt"]
    loaded_count = 0

    for story_file in story_files:
        filepath = data_dir / story_file
        if filepath.exists():
            print(f"Loading {story_file}...")
            documents = document_loader.load_single_document(str(filepath))
            if documents:
                retriever.add_document(str(filepath), documents)
                loaded_count += 1
                print(f"✅ Loaded {len(documents)} chunks from {story_file}")
            else:
                print(f"❌ Failed to load {story_file}")
        else:
            print(f"⚠️  {story_file} not found")

    if loaded_count == 0:
        print("❌ No documents loaded - cannot test retrieval")
        return False

    # Test queries - mix of simple and complex
    test_queries = [
        # Simple queries (should use sentence window)
        "What is the horse's name?",
        "Where does the frog live?",

        # Complex queries (should use auto-merging)
        "Why do you think the frog is always jumping around?",
        "What might the horse be dreaming about at the end?",
        "Compare the personalities of the frog and horse",
        "How does the frog's adventurous nature affect his experiences?"
    ]

    print(f"\n🧪 Testing {len(test_queries)} queries...")
    print("-" * 40)

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")

        try:
            # Get retrieval results
            retrieved_docs = retriever.retrieve_top_documents(query, top_k=3)

            if retrieved_docs:
                print(f"   ✅ Retrieved {len(retrieved_docs)} results")

                # Show sources
                sources = set()
                for doc in retrieved_docs:
                    filename = doc.get('metadata', {}).get('filename', 'Unknown')
                    sources.add(filename)

                print(f"   📄 Sources: {', '.join(sources)}")

                # Show first result preview
                if retrieved_docs:
                    first_doc = retrieved_docs[0]['document'][:100]
                    similarity = retrieved_docs[0].get('similarity', 0)
                    print(".3f")
                    print(f"      \"{first_doc}...\"")

                results.append({
                    'query': query,
                    'success': True,
                    'results_count': len(retrieved_docs),
                    'sources': list(sources)
                })
            else:
                print("   ❌ No results retrieved")
                results.append({
                    'query': query,
                    'success': False,
                    'results_count': 0,
                    'sources': []
                })

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'query': query,
                'success': False,
                'error': str(e),
                'results_count': 0,
                'sources': []
            })

    # Summary
    print("\n📊 SUMMARY")
    print("-" * 20)

    successful = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"Total queries: {total}")
    print(f"Successful: {successful}")
    print(".1f")

    # Show which method was likely used for each query
    print("\n🔍 Retrieval Method Analysis:")
    print("-" * 30)

    for result in results:
        query = result['query']
        is_complex = any(word in query.lower() for word in [
            'why', 'how', 'explain', 'compare', 'analyze',
            'relationship', 'difference', 'benefits', 'impact'
        ]) or len(query.split()) > 12

        method = "Auto-merging" if is_complex else "Sentence window"
        status = "✅" if result['success'] else "❌"

        print(f"{status} {method}: {query[:50]}...")

    print("\n🎉 Dual retrieval system test completed!")
    return successful > 0

if __name__ == "__main__":
    success = test_dual_retrieval()
    sys.exit(0 if success else 1)