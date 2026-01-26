import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.retrieval.bm25_sqlite_store import BM25SqliteStore
from cubo.retrieval.vector_store import FaissStore


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


class TestStreamingFaiss(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "documents.db"

        # Create a real SQLite DB with vectors for the store to read
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("CREATE TABLE vectors (id TEXT PRIMARY KEY, vector BLOB)")
            # Insert 200 dummy vectors
            # vector format in DB depending on FaissStore is usually serialized numpy or similar
            # But FaissStore.get_vectors usually decodes it.
            # Here we mock get_vectors, so we just need IDs in DB.
            conn.executemany(
                "INSERT INTO vectors (id, vector) VALUES (?, ?)",
                [(str(i), b"dummy") for i in range(200)],
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_start_rebuild_calls(self):
        # We verify that _rebuild_index_from_db calls train() and add_batch()

        mock_index_manager = MagicMock()
        mock_index_manager.hot_fraction = 0.5  # 100 hot, 100 cold

        store = FaissStore(db_path=self.db_path, index_dir=Path(self.test_dir), dimension=10)
        store._index = mock_index_manager  # Inject mock

        # Mock get_vectors to return dummy lists
        store.get_vectors = MagicMock(side_effect=lambda ids: {did: [0.1] * 10 for did in ids})

        # Run rebuild
        store._rebuild_index_from_db()

        # Verify train was called (sample size logic might limit it, but with 200 it should fetch all)
        # sample_size=50000, so it fetches all 200 for training
        self.assertTrue(mock_index_manager.train.called)
        args, _ = mock_index_manager.train.call_args
        self.assertEqual(len(args[0]), 200)  # Should train on all available if < sample

        # Verify add_batch was called
        # Default batch_size=5000, total 200. Should be 1 batch.
        # But logic splits batch hot/cold.
        # Hot fraction 0.5 -> 100 hot.
        # Batch 0-200. Hot cap 100.
        # Logic should call add_batch twice (hot part, cold part) OR logic inside handles single call?
        # My implementation loops and splits:
        # hot_v = batch[:100], cold_v = batch[100:]
        # add_batch(hot), add_batch(cold)

        # Check calls to add_batch
        self.assertGreaterEqual(mock_index_manager.add_batch.call_count, 2)

        # Verify save called
        self.assertTrue(mock_index_manager.save.called)

    def test_faiss_manager_structure(self):
        """Verify FAISSIndexManager has the new methods."""
        manager = FAISSIndexManager(dimension=8, nlist=10)
        self.assertTrue(hasattr(manager, "train"))
        self.assertTrue(hasattr(manager, "add_batch"))
        self.assertTrue(hasattr(manager, "reset"))


if __name__ == "__main__":
    unittest.main()
