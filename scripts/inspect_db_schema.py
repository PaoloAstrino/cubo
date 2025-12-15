#!/usr/bin/env python
"""Inspect documents.db schema and sample data"""
import sqlite3
import sys

db_path = 'results/tonight_full/storage/documents.db'
print(f"DB: {db_path}\n")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Show schema
print("=== SCHEMA ===")
c.execute("SELECT sql FROM sqlite_master WHERE type='table'")
for row in c.fetchall():
    print(row[0])
    print()

# Show counts
print("\n=== COUNTS ===")
for table in ['documents', 'collections', 'collection_documents', 'vectors']:
    try:
        c.execute(f'SELECT count(*) FROM {table}')
        print(f"{table}: {c.fetchone()[0]}")
    except Exception as e:
        print(f"{table}: ERROR - {e}")

# Sample documents
print("\n=== SAMPLE DOCUMENTS ===")
c.execute('SELECT id, content, metadata FROM documents LIMIT 3')
rows = c.fetchall()
for row in rows:
    doc_id, content, metadata = row
    print(f"id: {doc_id}")
    print(f"content (first 200 chars): {content[:200] if content else '(empty)'}")
    print(f"metadata: {metadata}")
    print()

# Sample vectors
print("=== SAMPLE VECTORS ===")
c.execute('SELECT id, dtype, dim FROM vectors LIMIT 3')
rows = c.fetchall()
for row in rows:
    vec_id, dtype, dim = row
    print(f"id: {vec_id}")
    print(f"dtype: {dtype}")
    print(f"dim: {dim}")
    print()

conn.close()
