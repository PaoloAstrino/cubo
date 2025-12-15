#!/usr/bin/env python
"""Comprehensive diagnostic to find the ID mismatch issue"""
import sqlite3
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.config import config
from cubo.indexing.faiss_index import FAISSIndexManager

index_dir = Path("results/tonight_full/storage")
db_path = index_dir / "documents.db"
metadata_path = index_dir / "metadata.json"

print("="*80)
print("COMPREHENSIVE DIAGNOSTIC")
print("="*80)

# 1. Check metadata
print("\n[1] METADATA CHECK")
import json
with open(metadata_path, 'r') as f:
    meta = json.load(f)
print(f"hot_ids count: {len(meta['hot_ids'])}")
print(f"cold_ids count: {len(meta['cold_ids'])}")
print(f"First 5 hot IDs: {meta['hot_ids'][:5]}")

# 2. Check documents.db
print("\n[2] DOCUMENTS.DB CHECK")
conn = sqlite3.connect(str(db_path))
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM documents")
doc_count = c.fetchone()[0]
print(f"Documents in DB: {doc_count}")

c.execute("SELECT id FROM documents LIMIT 5")
db_ids = [row[0] for row in c.fetchall()]
print(f"First 5 DB IDs: {db_ids}")

# 3. Check if metadata IDs exist in DB
print("\n[3] ID CROSS-CHECK")
sample_meta_ids = meta['hot_ids'][:10]
for mid in sample_meta_ids:
    c.execute("SELECT COUNT(*) FROM documents WHERE id=?", (mid,))
    exists = c.fetchone()[0]
    print(f"ID {mid[:50]}: {'EXISTS' if exists else 'MISSING'} in DB")

# 4. Load FAISS index and test search
print("\n[4] FAISS INDEX TEST")
try:
    faiss_mgr = FAISSIndexManager(
        dimension=768,
        index_dir=index_dir
    )
    faiss_mgr.load()
    print(f"FAISS hot index loaded: {faiss_mgr.hot_index is not None}")
    print(f"FAISS cold index loaded: {faiss_mgr.cold_index is not None}")
    print(f"hot_ids count: {len(faiss_mgr.hot_ids)}")
    print(f"cold_ids count: {len(faiss_mgr.cold_ids)}")
    
    # Test search
    import numpy as np
    test_vec = np.random.rand(768).astype('float32')
    results = faiss_mgr.search(test_vec, k=3)
    print(f"\nSearch returned {len(results)} results:")
    for i, res in enumerate(results):
        rid = res['id']
        c.execute("SELECT COUNT(*) FROM documents WHERE id=?", (rid,))
        exists = c.fetchone()[0]
        print(f"  Result {i+1}: ID={rid[:50]}, distance={res['distance']:.4f}, IN_DB={'YES' if exists else 'NO'}")
except Exception as e:
    print(f"ERROR loading/testing FAISS: {e}")

conn.close()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
