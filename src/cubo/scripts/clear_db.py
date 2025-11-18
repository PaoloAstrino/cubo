"""
Clear ChromaDB and test fresh upload with fixed embeddings.
"""
import shutil
import os
from pathlib import Path

print("=" * 80)
print("CLEARING CHROMADB")
print("=" * 80)

chroma_path = Path("./chroma_db")
if chroma_path.exists():
    print(f"\nDeleting {chroma_path}...")
    shutil.rmtree(chroma_path)
    print("✓ ChromaDB cleared!")
else:
    print(f"\n{chroma_path} doesn't exist, nothing to clear")

print("\n" + "=" * 80)
print("DATABASE CLEARED - Please re-upload your documents in the GUI")
print("=" * 80)
print("\nThe fixed code will now:")
print("1. Generate proper embeddings for sentence window chunks")
print("2. Generate proper embeddings for auto-merging chunks")
print("3. Store them correctly in ChromaDB")
print("\nAfter re-uploading, retrieval should work correctly!")
