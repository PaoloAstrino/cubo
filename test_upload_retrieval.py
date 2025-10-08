"""
Test script: Upload all animal stories and test retrieval.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.model_loader import ModelManager
from src.retriever import DocumentRetriever
from src.document_loader import DocumentLoader
from src.logger import logger

def test_upload_and_retrieval():
    """Upload documents and test retrieval."""
    print("=" * 80)
    print("UPLOAD AND RETRIEVAL TEST")
    print("=" * 80)

    # Load model
    print("\n1. Loading embedding model...")
    model_manager = ModelManager()
    model = model_manager.load_model()
    print(f"   [OK] Model loaded")

    # Initialize retriever and document loader
    print("\n2. Initializing retriever...")
    retriever = DocumentRetriever(
        model=model,
        use_sentence_window=True,
        use_auto_merging=True
    )
    doc_loader = DocumentLoader()
    print(f"   [OK] Retriever initialized")

    # Upload all animal story files
    print("\n3. Uploading animal story files...")
    data_dir = Path("data")
    animal_stories = [
        "cat_story.txt",
        "dog_story.txt",
        "elephant_story.txt",
        "frog_story.txt",
        "horse_story.txt",
        "lion_story.txt",
        "rabbit_story.txt"
    ]

    uploaded_count = 0
    for filename in animal_stories:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                print(f"   Uploading {filename}...", end=" ")
                chunks = doc_loader.load_single_document(str(filepath))
                success = retriever.add_document(str(filepath), chunks)
                if success:
                    print(f"[OK] ({len(chunks)} chunks)")
                    uploaded_count += 1
                else:
                    print("[!] Already exists")
            except Exception as e:
                print(f"[X] Error: {e}")
        else:
            print(f"   [X] {filename} not found")

    print(f"\n   Total uploaded: {uploaded_count}/{len(animal_stories)} files")

    # Check what's in the database
    print("\n4. Verifying database contents...")
    all_data = retriever.collection.get()
    if all_data and all_data.get('metadatas'):
        from collections import Counter
        file_counts = Counter(m.get('filename', 'Unknown') for m in all_data['metadatas'])
        print(f"   Files in database: {len(file_counts)}")
        for filename, count in sorted(file_counts.items()):
            print(f"     - {filename}: {count} chunks")

    # Test queries
    test_queries = [
        ("tell me about the frog", "frog_story.txt"),
        ("tell me about the horse", "horse_story.txt"),
        ("tell me about the lion", "lion_story.txt"),
        ("what is a cat", "cat_story.txt"),
        ("describe the elephant", "elephant_story.txt"),
        ("tell me about the dog", "dog_story.txt"),
        ("what is a rabbit", "rabbit_story.txt"),
    ]

    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL")
    print("=" * 80)

    correct_retrievals = 0
    total_queries = len(test_queries)

    for query, expected_file in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"Expected: {expected_file}")
        print(f"{'='*80}")

        try:
            results = retriever.retrieve_top_documents(query, top_k=6)

            if results:
                print(f"\nRetrieved {len(results)} results:")
                retrieved_files = []
                for i, result in enumerate(results[:6], 1):
                    filename = result.get('metadata', {}).get('filename', 'Unknown')
                    similarity = result.get('similarity', 0)
                    doc_preview = result.get('document', '')[:80].replace('\n', ' ')
                    print(f"  {i}. {filename} (sim: {similarity:.4f})")
                    print(f"     {doc_preview}...")
                    if filename not in retrieved_files:
                        retrieved_files.append(filename)

                # Check if expected file is in top result
                top_file = results[0].get('metadata', {}).get('filename', 'Unknown')
                if expected_file in top_file:
                    print(f"\n  [OK] CORRECT: Top result matches expected file!")
                    correct_retrievals += 1
                else:
                    print(f"\n  [X] INCORRECT: Expected {expected_file} but got {top_file}")
                    print(f"     Files retrieved: {retrieved_files}")
            else:
                print("  [X] No results returned!")

        except Exception as e:
            print(f"  [X] Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    accuracy = (correct_retrievals / total_queries * 100) if total_queries > 0 else 0
    print(f"Correct retrievals: {correct_retrievals}/{total_queries} ({accuracy:.1f}%)")

    if accuracy >= 80:
        print("[OK] Retrieval is working well!")
    elif accuracy >= 50:
        print("[!] Retrieval is partially working, needs improvement")
    else:
        print("[X] Retrieval needs significant fixes")

if __name__ == "__main__":
    test_upload_and_retrieval()
