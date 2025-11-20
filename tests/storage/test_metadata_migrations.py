import sqlite3
from pathlib import Path

from src.cubo.storage.metadata_manager import MetadataManager, get_metadata_manager


def test_migrate_scaffold_mappings_add_run_id(tmp_path: Path):
    db_path = tmp_path / "old_metadata.db"
    # Create an old schema DB: scaffold_mappings without run_id column
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute('''CREATE TABLE scaffold_mappings (scaffold_id TEXT, chunk_id TEXT, metadata TEXT, PRIMARY KEY (scaffold_id, chunk_id))''')
    cur.execute('''INSERT INTO scaffold_mappings (scaffold_id, chunk_id, metadata) VALUES ('s1', 'c1', '{}')''')
    conn.commit()
    conn.close()

    # Now instantiate MetadataManager pointing at this DB and allow migration
    mgr = MetadataManager(db_path=str(db_path))

    # Query the table schema to confirm run_id column exists
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(scaffold_mappings)")
    cols = [r[1] for r in cur.fetchall()]
    assert 'run_id' in cols
    assert 'scaffold_id' in cols
    assert 'chunk_id' in cols
    assert 'metadata' in cols

    # Verify the existing data has been copied with default empty run_id
    cur.execute("SELECT run_id, scaffold_id, chunk_id FROM scaffold_mappings WHERE scaffold_id='s1' and chunk_id='c1'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == ''  # run_id default was added as empty string
    conn.close()
