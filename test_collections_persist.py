#!/usr/bin/env python3
"""Test script to verify collections persistence in SQLite."""
import sqlite3
from pathlib import Path

db_path = Path("./data/cubo_index/documents.db")

print(f"Checking database at: {db_path}")
print(f"Database exists: {db_path.exists()}")
print()

if db_path.exists():
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        # Check if tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[t[0] for t in tables]}")
        print()

        # Check collections
        try:
            cursor = conn.execute("SELECT * FROM collections")
            collections = cursor.fetchall()
            print(f"Number of collections: {len(collections)}")
            for coll in collections:
                print(f"  - ID: {coll['id']}, Name: {coll['name']}, Documents: ?")
            print()

            # Count documents per collection
            cursor = conn.execute(
                """
                SELECT c.id, c.name, COUNT(cd.document_id) as doc_count
                FROM collections c
                LEFT JOIN collection_documents cd ON c.id = cd.collection_id
                GROUP BY c.id
            """
            )
            results = cursor.fetchall()
            print("Collections with document counts:")
            for row in results:
                print(f"  - {row['name']}: {row['doc_count']} documents")
        except Exception as e:
            print(f"Error querying collections: {e}")
else:
    print("Database file does not exist!")
