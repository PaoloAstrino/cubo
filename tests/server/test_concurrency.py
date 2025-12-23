import threading
import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_concurrent_build_and_query():
    """Run build-index concurrently with queries and assert no 500s.

    This test simulates a long-running build to ensure the query endpoint remains
    responsive (returns 503 or 200 but not 500) and does not crash under concurrency.
    """
    mock_app = MagicMock()
    mock_app.retriever = MagicMock()
    # Start with no docs
    mock_app.retriever.collection = MagicMock()
    mock_app.retriever.collection.count.return_value = 0
    mock_app.retriever.retrieve_top_documents.return_value = []
    mock_app.generator = MagicMock()
    mock_app.generate_response_safe.return_value = "Generated"

    def fake_build_index():
        # Simulate long build, then set retriever count > 0
        time.sleep(0.5)
        mock_app.retriever.collection.count.return_value = 5
        return 3

    mock_app.build_index.side_effect = fake_build_index

    with patch("cubo.server.api.cubo_app", mock_app):
        with patch("cubo.server.api.security_manager") as mock_security:
            mock_security.scrub.side_effect = lambda x: x
            from cubo.server.api import app

            client = TestClient(app)

            # Start build in background thread
            t = threading.Thread(
                target=lambda: client.post("/api/build-index", json={"force_rebuild": True}),
            )
            t.start()

            # While build runs, perform multiple queries; expect only 503/200 during build
            statuses = []
            for _ in range(5):
                resp = client.post("/api/query", json={"query": "test", "top_k": 1})
                statuses.append(resp.status_code)
                time.sleep(0.05)

            t.join()

            # Ensure none of the calls returned 500 and all are in expected set
            assert not any(s == 500 for s in statuses)
            assert set(statuses) <= {200, 503}, f"Unexpected statuses during build: {statuses}"

            # After build completes, queries should succeed (200)
            post_build = client.post("/api/query", json={"query": "test", "top_k": 1})
            assert post_build.status_code == 200


def test_build_index_crash_while_adding(tmp_path):
    """Simulate an error during build_index and verify no partial SQLite commit.
    This creates a temporary FaissStore in memory and simulates commit failure.
    """
    import sqlite3
    from pathlib import Path

    from cubo.retrieval.vector_store import FaissStore

    tmpdir = tmp_path / "faiss_temp"
    tmpdir.mkdir()
    store = FaissStore(64, index_dir=Path(tmpdir))
    # Reset store to ensure clean DB
    store.reset()

    # Patch sqlite3.connect to raise on commit only for the add operations
    original_connect = sqlite3.connect

    def faulty_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)

        class Proxy:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, *a, **kw):
                return self._conn.execute(*a, **kw)

            def executemany(self, *a, **kw):
                return self._conn.executemany(*a, **kw)

            def commit(self):
                # Simulate failure
                raise sqlite3.OperationalError("Simulated commit failure")

            def rollback(self):
                return self._conn.rollback()

            def close(self):
                return self._conn.close()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                try:
                    self._conn.__exit__(exc_type, exc, tb)
                except Exception:
                    pass
                return False

            def __getattr__(self, name):
                return getattr(self._conn, name)

        return Proxy(conn)

    sqlite3.connect = faulty_connect

    try:
        import numpy as np

        embeddings = [np.zeros(64, dtype=np.float32)]
        docs = ["Test doc"]
        metas = [{"filename": "a.txt"}]
        with patch("cubo.utils.logger.logger"):
            try:
                store.add(embeddings=embeddings, documents=docs, metadatas=metas, ids=["d1"])
            except sqlite3.OperationalError:
                # Expected
                pass
            finally:
                # Restore sqlite connect now so further cleanup uses the real DB
                sqlite3.connect = original_connect

        # After failing add, there should be no vectors in SQLite DB
        assert store.count_vectors() == 0
    finally:
        # Restore sqlite connect first so cleanup uses real DB connections
        # Ensure connect is restored for cleanup (in case not restored above)
        sqlite3.connect = original_connect
        # Ensure store has expected attribute to avoid close errors
        if not hasattr(store, "_embeddings"):
            store._embeddings = {}
        # Attempt a proper close/reset and then remove dir; tolerate permission error on Windows
        try:
            store.reset()
        except Exception:
            pass
        try:
            store.close()
        except Exception:
            pass
        import shutil

        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
