from pathlib import Path
import sqlite3

import numpy as np

from cubo.retrieval.vector_store import FaissStore


def test_delete_removes_collection_links(tmp_path: Path):
    # Create a FAISS store in a temp directory
    store = FaissStore(dimension=4, index_dir=tmp_path)

    # Add a collection and a document
    coll = store.create_collection(name="test-coll")
    doc_id = "doc1"
    metadata = {"filename": "doc1.pdf"}

    # Add vector and document via store.add
    vec = np.zeros(4, dtype=np.float32)
    store.add(embeddings=[vec], documents=["hello"], metadatas=[metadata], ids=[doc_id])

    # Add document to collection
    res = store.add_documents_to_collection(coll["id"], [doc_id])
    assert res["added_count"] == 1

    # Ensure collection has document
    docs_in_coll = store.get_collection_documents(coll["id"])
    assert doc_id in docs_in_coll

    # Now delete the document
    store.delete(ids=[doc_id])

    # After deletion, collection should no longer include the doc
    docs_in_coll_after = store.get_collection_documents(coll["id"])
    assert doc_id not in docs_in_coll_after

    # And documents table should not contain the document
    assert store.count() == 0


def test_enqueue_deletion_removes_collection_links(tmp_path: Path):
    store = FaissStore(dimension=4, index_dir=tmp_path)
    coll = store.create_collection(name="col2")
    doc_id = "doc2"
    metadata = {"filename": "doc2.pdf"}

    vec = np.ones(4, dtype=np.float32)
    store.add(embeddings=[vec], documents=["world"], metadatas=[metadata], ids=[doc_id])
    store.add_documents_to_collection(coll["id"], [doc_id])

    # Enqueue deletion
    job_id = store.enqueue_deletion(doc_id, trace_id="t1", force=True)
    assert job_id is not None

    # The document should already be removed from the documents table (enqueue_deletion deletes DB rows)
    assert store.count() == 0

    # Collection should no longer include the doc
    docs_in_coll = store.get_collection_documents(coll["id"])
    assert doc_id not in docs_in_coll

    # Deletion job should be present in deletion_jobs table
    with sqlite3.connect(str(store._db_path)) as conn:
        row = conn.execute(
            "SELECT id, doc_id, status FROM deletion_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert row is not None
    assert row[1] == doc_id
