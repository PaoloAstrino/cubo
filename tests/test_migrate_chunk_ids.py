from pathlib import Path
import json
import tempfile
import os

import runpy
import subprocess
import sys
import tempfile
from src.config import config
from src.ingest.fast_pass_ingestor import FastPassIngestor
from src.ingest.deep_ingestor import DeepIngestor
from src.retriever import DocumentRetriever


def test_migrate_chunk_ids_dry_run(tmp_path: Path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Test doc for migration.")

    fast_out = tmp_path / "fast_out"
    FastPassIngestor(output_dir=str(fast_out), skip_model=True).ingest_folder(str(folder))

    # Ensure the fast pass created file_hash entries
    dp = DeepIngestor(input_folder=str(folder), output_dir=str(tmp_path / "deep_out"))
    dp.ingest()

    # Run migration script as a dry-run execution (we only verify that the module defines `main`)
    path = os.path.join(os.getcwd(), 'scripts', 'migrate_chunk_ids.py')
    assert os.path.exists(path)
    g = runpy.run_path(path)
    assert 'main' in g


def test_migrate_chunk_ids_apply(tmp_path: Path):
    # Setup a temporary chroma db path
    tmpdb = tmp_path / 'chroma'
    tmpdb.mkdir()
    # patch config to point to tmpdb
    from src.config import config as cfg
    cfg.set('chroma_db_path', str(tmpdb))

    # Create retriever and collection and add a couple of items with legacy filename-based ids
    retr = DocumentRetriever(model=None)
    coll = retr.client.get_or_create_collection('test_migrate')

    # Add two items with filename-based ids and metadata with file_hash
    ids = ['file1.txt_s0', 'file1.txt_s1']
    docs = ['doc1 text', 'doc2 text']
    embeddings = [[0.1]*32, [0.2]*32]
    metas = [
        {'filename': 'file1.txt', 'file_hash': 'abc123', 'chunk_index': 0, 'sentence_index': 0},
        {'filename': 'file1.txt', 'file_hash': 'abc123', 'chunk_index': 1, 'sentence_index': 1},
    ]
    coll.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    # Ensure old ids exist
    retrieved = coll.get(ids=ids)
    assert retrieved and retrieved.get('ids')

    # Run migration with --apply and --safe-apply (run in-process)
    import runpy
    old_argv = sys.argv
    try:
        sys.argv = ['migrate_chunk_ids.py', '--db-path', str(tmpdb), '--collection', 'test_migrate', '--apply', '--safe-apply']
        runpy.run_path(str(Path.cwd() / 'scripts' / 'migrate_chunk_ids.py'), run_name='__main__')
    finally:
        sys.argv = old_argv

    # New ids should be abc123_s0 and abc123_s1
    new_ids = ['abc123_s0', 'abc123_s1']
    new_retrieved = coll.get(ids=new_ids)
    assert new_retrieved and new_retrieved.get('ids')
    # Old ids should be removed
    old_retrieved = coll.get(ids=ids)
    if old_retrieved and old_retrieved.get('ids'):
        assert len(old_retrieved.get('ids')) == 0
