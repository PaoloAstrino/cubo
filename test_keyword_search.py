#!/usr/bin/env python3
"""
Test Keyword Search Implementation
Test the BM25-enhanced keyword search on animal stories.
"""

import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.retriever import DocumentRetriever
from src.document_loader import DocumentLoader
from sentence_transformers import SentenceTransformer
from src.logger import logger


def clear_database(retriever):
    """Clear the database for a fresh test."""
    print("\n=== Clearing Database ===")
    try:
        retriever.clear_documents()
        print("✓ Database cleared successfully")
        return True
    except Exception as e:
        print(f"✗ Error clearing database: {e}")
        return False


def upload_animal_stories(retriever, doc_loader):
    """Upload all animal story files."""
    print("\n=== Uploading Animal Stories ===")
    
    # List of animal story files
    animal_files = [
        "data/frog_story.txt",
        "data/lion_story.txt",
        "data/elephant_story.txt",
        "data/horse_story.txt",
        "data/cat_story.txt",
        "data/dog_story.txt",
        "data/rabbit_story.txt"
    ]
    
    # Check which files exist
    existing_files = [f for f in animal_files if os.path.exists(f)]
    missing_files = [f for f in animal_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠ Missing files: {', '.join(missing_files)}")
    
    print(f"Found {len(existing_files)} story files")
    
    # Upload each file
    successful = 0
    failed = 0
    
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        try:
            # Load document and get chunks (includes metadata)
            chunks = doc_loader.load_single_document(filepath)
            
            # Add to retriever
            result = retriever.add_document(filepath, chunks)
            if result:
                successful += 1
                print(f"  ✓ {filename} ({len(chunks)} chunks)")
            else:
                failed += 1
                print(f"  ✗ {filename} (add_document returned False)")
        except Exception as e:
            failed += 1
            print(f"  ✗ {filename}: {e}")
    
    print(f"\nUpload complete: {successful} successful, {failed} failed")
    return successful, failed


def test_keyword_retrieval(retriever):
    """Test keyword-based retrieval with specific animal queries."""
    print("\n=== Testing Keyword Retrieval ===")
    
    # Test queries - each should return the corresponding animal story
    test_cases = [
        ("Tell me about the frog", "frog_story.txt"),
        ("What happened with the lion?", "lion_story.txt"),
        ("Describe the elephant", "elephant_story.txt"),
        ("Tell me about the horse", "horse_story.txt"),
        ("What about the cat?", "cat_story.txt"),
        ("Tell me the dog story", "dog_story.txt"),
        ("Describe the rabbit", "rabbit_story.txt"),
    ]
    
    correct = 0
    total = len(test_cases)
    results = []
    
    for query, expected_file in test_cases:
        print(f"\n--- Query: '{query}' ---")
        print(f"Expected: {expected_file}")
        
        try:
            # Retrieve top documents
            docs = retriever.retrieve_top_documents(query, top_k=5)
            
            if not docs:
                print("✗ No results returned")
                results.append({
                    'query': query,
                    'expected': expected_file,
                    'got': 'NO_RESULTS',
                    'correct': False
                })
                continue
            
            # Check which files were returned
            returned_files = [doc.get('metadata', {}).get('filename', 'UNKNOWN') 
                            for doc in docs]
            
            # Display top results
            print(f"Top results:")
            for i, doc in enumerate(docs[:3], 1):
                filename = doc.get('metadata', {}).get('filename', 'UNKNOWN')
                similarity = doc.get('similarity', 0.0)
                base_sim = doc.get('base_similarity', similarity)
                print(f"  {i}. {filename}")
                print(f"     Semantic: {base_sim:.4f}, Combined: {similarity:.4f}")
                if 'document' in doc:
                    preview = doc['document'][:100].replace('\n', ' ')
                    print(f"     Preview: {preview}...")
            
            # Check if expected file is in top result
            top_file = returned_files[0] if returned_files else 'NO_FILE'
            is_correct = (top_file == expected_file)
            
            if is_correct:
                print(f"✓ CORRECT - Top result is {expected_file}")
                correct += 1
            else:
                print(f"✗ INCORRECT - Expected {expected_file}, got {top_file}")
            
            results.append({
                'query': query,
                'expected': expected_file,
                'got': top_file,
                'correct': is_correct,
                'all_results': returned_files[:3]
            })
            
        except Exception as e:
            print(f"✗ Error during retrieval: {e}")
            results.append({
                'query': query,
                'expected': expected_file,
                'got': f'ERROR: {e}',
                'correct': False
            })
    
    # Summary
    print("\n" + "="*70)
    print(f"RESULTS SUMMARY")
    print("="*70)
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Correct: {correct}/{total} ({accuracy:.1f}%)")
    print(f"Incorrect: {total - correct}/{total}")
    
    # Detailed results
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result['correct'] else "✗"
        print(f"\n{i}. {status} Query: '{result['query']}'")
        print(f"   Expected: {result['expected']}")
        print(f"   Got: {result['got']}")
        if 'all_results' in result and not result['correct']:
            print(f"   Top 3: {', '.join(result['all_results'])}")
    
    return accuracy, results


def test_bm25_scoring_detail(retriever):
    """Test BM25 scoring in detail for a specific query."""
    print("\n=== Detailed BM25 Scoring Test ===")
    
    query = "Tell me about the frog"
    print(f"Query: '{query}'")
    
    # Enable debug logging temporarily
    import logging
    logger.setLevel(logging.DEBUG)
    
    print("\nRetrieving with BM25 keyword scoring enabled...")
    docs = retriever.retrieve_top_documents(query, top_k=7)
    
    # Reset logging
    logger.setLevel(logging.INFO)
    
    if not docs:
        print("No results returned")
        return
    
    print(f"\nAll {len(docs)} results:")
    for i, doc in enumerate(docs, 1):
        filename = doc.get('metadata', {}).get('filename', 'UNKNOWN')
        similarity = doc.get('similarity', 0.0)
        base_sim = doc.get('base_similarity', similarity)
        boost = similarity - base_sim
        
        print(f"\n{i}. {filename}")
        print(f"   Semantic similarity: {base_sim:.4f}")
        print(f"   Combined score: {similarity:.4f}")
        print(f"   Keyword boost: {boost:+.4f}")
        
        # Show a preview
        if 'document' in doc:
            preview = doc['document'][:150].replace('\n', ' ')
            print(f"   Preview: {preview}...")


def main():
    """Main test function."""
    print("="*70)
    print("KEYWORD SEARCH TEST - BM25 Enhanced Retrieval")
    print("="*70)
    
    # Initialize model and retriever
    print("\n=== Initializing ===")
    try:
        print("Loading embedding model...")
        model = SentenceTransformer('models/embeddinggemma-300m')
        print("✓ Model loaded")
        
        print("Initializing document loader...")
        doc_loader = DocumentLoader()
        print("✓ Document loader initialized")
        
        print("Initializing retriever...")
        retriever = DocumentRetriever(
            model=model,
            use_sentence_window=True,
            use_auto_merging=False  # Disable to test pure BM25+semantic hybrid
        )
        print("✓ Retriever initialized")
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return 1
    
    # Clear database
    print("\n=== Clearing Database ===")
    try:
        # Get all document IDs and delete them
        all_docs = retriever.collection.get()
        if all_docs['ids']:
            retriever.collection.delete(ids=all_docs['ids'])
            print(f"✓ Deleted {len(all_docs['ids'])} existing documents")
        retriever.current_documents.clear()
        # Clear BM25 statistics
        retriever.doc_term_freq.clear()
        retriever.doc_lengths.clear()
        retriever.term_doc_freq.clear()
        retriever.avg_doc_length = 0
        print("✓ Database cleared successfully")
    except Exception as e:
        print(f"⚠ Error clearing database: {e}")
        print("Continuing anyway...")
    
    successful, failed = upload_animal_stories(retriever, doc_loader)
    if successful == 0:
        print("No documents uploaded successfully. Aborting test.")
        return 1
    
    # Wait a moment for processing
    time.sleep(1)
    
    # Test keyword retrieval
    accuracy, results = test_keyword_retrieval(retriever)
    
    # Detailed BM25 test
    test_bm25_scoring_detail(retriever)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"Final Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("✓ PASSED - Keyword search working well (≥80% accuracy)")
        return 0
    elif accuracy >= 50:
        print("⚠ PARTIAL - Keyword search needs improvement (50-80% accuracy)")
        return 0
    else:
        print("✗ FAILED - Keyword search not working (<50% accuracy)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
