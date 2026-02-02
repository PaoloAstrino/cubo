import hashlib
import sqlite3
import threading

import pytest

from cubo.ingestion.document_loader import DocumentLoader
from cubo.retrieval.vector_store import FaissStore
from cubo.storage.metadata_manager import MetadataManager
from cubo.utils.utils import Utils, resolve_hf_revision


def test_compute_file_hash_is_sha256(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("hello world")
    loader = DocumentLoader()
    h = loader._compute_file_hash(str(p))
    # compare with hashlib.sha256
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == expected


def test_faissstore_validate_ids_accepts_and_rejects():
    store = FaissStore.__new__(FaissStore)
    # should accept safe ids
    store._validate_ids(["abc123", "uuid-1_2.3:4"])
    # should reject malicious ids
    with pytest.raises(ValueError):
        store._validate_ids(["good", "bad; DROP TABLE users;"])


def test_metadata_manager_whitelist_fields():
    mgr = MetadataManager.__new__(MetadataManager)
    # minimal setup for DB and lock
    mgr.db_path = ":memory:"
    mgr.conn = sqlite3.connect(mgr.db_path)
    mgr._lock = threading.Lock()
    cur = mgr.conn.cursor()
    # create ingestion_runs table
    cur.execute(
        """
        CREATE TABLE ingestion_runs (
            id TEXT PRIMARY KEY,
            chunks_count INTEGER,
            output_parquet TEXT,
            status TEXT,
            finished_at TEXT
        )
        """
    )
    cur.execute("INSERT INTO ingestion_runs (id, status) VALUES (?, ?)", ("run1", "started"))
    mgr.conn.commit()

    # valid update works and persists
    mgr.update_ingestion_run_details("run1", chunks_count=5, status="ok")
    cur.execute("SELECT chunks_count, status FROM ingestion_runs WHERE id = ?", ("run1",))
    row = cur.fetchone()
    assert row[0] == 5
    assert row[1] == "ok"


def test_hf_revision_resolver_requires_pin(monkeypatch):
    # ensure no env vars are set
    monkeypatch.delenv("HF_PINNED_REVISION", raising=False)
    monkeypatch.delenv("HF_ALLOW_UNPINNED_HF_DOWNLOADS", raising=False)
    with pytest.raises(RuntimeError):
        resolve_hf_revision()

    # set pin and ensure it returns
    monkeypatch.setenv("HF_PINNED_REVISION", "v1.2.3")
    assert resolve_hf_revision() == "v1.2.3"


def test_create_sentence_window_chunks_handles_unpinned_tokenizer(monkeypatch):
    # ensure env not set to allow unpinned
    monkeypatch.delenv("HF_PINNED_REVISION", raising=False)
    monkeypatch.delenv("HF_ALLOW_UNPINNED_HF_DOWNLOADS", raising=False)

    # Using a remote tokenizer name should not raise (exception is caught internally)
    chunks = Utils.create_sentence_window_chunks(
        "This is a sentence. Another.", tokenizer_name="some-remote/tokenizer"
    )
    assert isinstance(chunks, list)
    assert len(chunks) > 0
