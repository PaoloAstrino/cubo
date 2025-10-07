"""
Inspect what text is actually stored in ChromaDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from collections import defaultdict

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="cubo_documents")

# Get all data
all_data = collection.get()

print("=" * 80)
print("CHROMADB CONTENT INSPECTION")
print("=" * 80)

if not all_data or not all_data.get('documents'):
    print("\nNo data in collection")
else:
    # Group by filename
    file_chunks = defaultdict(list)
    for doc, metadata, doc_id in zip(all_data['documents'], all_data['metadatas'], all_data['ids']):
        filename = metadata.get('filename', 'Unknown')
        file_chunks[filename].append({
            'id': doc_id,
            'text': doc,
            'metadata': metadata
        })
    
    # Show first 3 chunks from each file
    for filename in sorted(file_chunks.keys()):
        chunks = file_chunks[filename]
        print(f"\n{'='*80}")
        print(f"File: {filename} ({len(chunks)} chunks)")
        print(f"{'='*80}")
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} (ID: {chunk['id']}):")
            print(f"Text length: {len(chunk['text'])} chars")
            print(f"First 200 chars: {chunk['text'][:200]}")
            print(f"Last 100 chars: ...{chunk['text'][-100:]}")
            
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")
