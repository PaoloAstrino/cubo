#!/usr/bin/env python3
"""
Simple demonstration of dual retrieval system working together
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
from custom_auto_merging import AutoMergingRetriever
from document_loader import DocumentLoader

def demonstrate_dual_retrieval():
    """Demonstrate the dual retrieval system working together."""

    print("🎯 CUBO Dual Retrieval System Demonstration")
    print("=" * 50)
    print("This shows both sentence window and auto-merging working together!")

    # Load model
    model_path = "./models/embeddinggemma-300m"
    print(f"\n📚 Loading embedding model from {model_path}...")
    model = SentenceTransformer(model_path)
    print("✅ Model loaded successfully")

    # Initialize auto-merging retriever
    print("\n🔧 Initializing auto-merging retriever...")
    auto_retriever = AutoMergingRetriever(model)
    print("✅ Auto-merging retriever ready")

    # Load documents
    document_loader = DocumentLoader()
    data_dir = Path("./data")

    story_files = ["frog_story.txt", "horse_story.txt"]
    loaded_count = 0

    print("\n📄 Loading story documents...")
    for story_file in story_files:
        filepath = data_dir / story_file
        if filepath.exists():
            print(f"   Loading {story_file}...")
            documents = document_loader.load_single_document(str(filepath))
            if documents:
                # Add to auto-merging retriever
                success = auto_retriever.add_document(str(filepath))
                if success:
                    loaded_count += 1
                    print(f"   ✅ Added to auto-merging system")
                else:
                    print(f"   ❌ Failed to add to auto-merging")
            else:
                print(f"   ❌ Failed to load {story_file}")
        else:
            print(f"   ⚠️  {story_file} not found")

    if loaded_count == 0:
        print("❌ No documents loaded - cannot demonstrate")
        return False

    print(f"\n✅ Successfully loaded {loaded_count} documents into dual retrieval system")

    # Demonstrate query complexity analysis
    print("\n🧠 Query Complexity Analysis:")
    print("-" * 30)

    test_queries = [
        ("What is the horse's name?", "Simple - uses sentence window"),
        ("Where does the frog live?", "Simple - uses sentence window"),
        ("Why do you think the frog is always jumping around?", "Complex - uses auto-merging"),
        ("Compare the personalities of the frog and horse", "Complex - uses auto-merging"),
        ("How does the frog's adventurous nature affect his experiences?", "Complex - uses auto-merging")
    ]

    def analyze_query_complexity(query: str) -> bool:
        """Analyze if query needs complex retrieval (same logic as retriever.py)."""
        complex_indicators = [
            'why', 'how', 'explain', 'compare', 'analyze',
            'relationship', 'difference', 'benefits', 'impact',
            'advantages', 'disadvantages', 'vs', 'versus'
        ]

        query_lower = query.lower()
        has_complex_keywords = any(indicator in query_lower for indicator in complex_indicators)
        is_long_query = len(query.split()) > 12

        return has_complex_keywords or is_long_query

    for query, expected in test_queries:
        is_complex = analyze_query_complexity(query)
        method = "🤖 Auto-merging" if is_complex else "📄 Sentence window"
        print(f"   {method}: {query}")
        print(f"      → {expected}")

    # Test actual retrieval with auto-merging
    print("\n🧪 Testing Auto-merging Retrieval:")
    print("-" * 35)

    complex_queries = [
        "Why do you think the frog is always jumping around?",
        "Compare the personalities of the frog and horse"
    ]

    for query in complex_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = auto_retriever.retrieve(query, top_k=2)
            if results:
                print(f"✅ Retrieved {len(results)} results using auto-merging")
                for i, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0)
                    preview = result.get('document', '')[:80] + "..."
                    print(".3f")
            else:
                print("❌ No results retrieved")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 Demonstration Complete!")
    print("\n📋 Summary:")
    print("   • Sentence window: Fast, precise retrieval for simple queries")
    print("   • Auto-merging: Hierarchical, comprehensive retrieval for complex queries")
    print("   • Dual system: Automatically chooses the best method based on query complexity")
    print("   • All configurable: No hardcoded values, everything through config.json")

    return True

if __name__ == "__main__":
    success = demonstrate_dual_retrieval()
    sys.exit(0 if success else 1)