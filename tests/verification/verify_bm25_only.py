
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Only import BM25 store
from cubo.retrieval.bm25_sqlite_store import BM25SqliteStore

class TestBM25SqliteStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = BM25SqliteStore(index_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_add_and_search(self):
        docs = [
            {"doc_id": "1", "text": "apple banana cherry", "metadata": {"type": "fruit"}},
            {"doc_id": "2", "text": "dog cat mouse", "metadata": {"type": "animal"}},
            {"doc_id": "3", "text": "apple pie recipe", "metadata": {"type": "food"}},
        ]
        self.store.index_documents(docs)

        # Search for "apple"
        results = self.store.search("apple")
        self.assertEqual(len(results), 2)
        ids = sorted([r["doc_id"] for r in results])
        self.assertEqual(ids, ["1", "3"])
        
        # Search for "cat"
        results = self.store.search("cat")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "2")
        
        # Verify persistence
        self.store = BM25SqliteStore(index_dir=self.test_dir)
        results = self.store.search("mouse")
        self.assertEqual(len(results), 1)

    def test_incremental_add(self):
        self.store.index_documents([{"doc_id": "1", "text": "apple"}])
        self.store.add_documents([{"doc_id": "2", "text": "banana"}])
        
        results = self.store.search("banana")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "2")

    def test_compute_score(self):
        self.store.index_documents([{"doc_id": "1", "text": "apple banana"}])
        
        # Score calculation
        score = self.store.compute_score(["apple"], "1")
        self.assertGreater(score, 0.0)
        
        score_zero = self.store.compute_score(["orange"], "1")
        self.assertEqual(score_zero, 0.0)

if __name__ == "__main__":
    unittest.main()
