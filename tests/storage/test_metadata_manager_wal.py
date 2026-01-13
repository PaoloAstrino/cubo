import sqlite3
import pytest
from pathlib import Path
from cubo.storage.metadata_manager import MetadataManager

def test_metadata_manager_wal_mode(tmp_path):
    """
    Verify that the MetadataManager correctly enables WAL mode and NORMAL synchronous setting.
    """
    db_file = tmp_path / "test_metadata.db"
    
    # Initialize manager
    manager = MetadataManager(db_path=str(db_file))
    
    try:
        # Query internal PRAGMA settings
        cur = manager.conn.cursor()
        
        # Check journal mode
        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL mode, got {journal_mode}"
        
        # Check synchronous setting
        # 1 = NORMAL, 2 = FULL (default), 0 = OFF
        cur.execute("PRAGMA synchronous;")
        sync_mode = cur.fetchone()[0]
        assert sync_mode == 1, f"Expected synchronous=NORMAL (1), got {sync_mode}"
        
        print(f"Verified: SQLite WAL mode enabled, Synchronous={sync_mode}")
        
    finally:
        # Close connection to allow file cleanup on Windows
        manager.conn.close()

def test_metadata_manager_concurrency(tmp_path):
    """
    Simple verification that multiple connections can exist (enabled by WAL).
    """
    db_file = tmp_path / "concurrency_test.db"
    manager = MetadataManager(db_path=str(db_file))
    
    try:
        # Create a second raw connection to the same file
        conn2 = sqlite3.connect(str(db_file))
        cur2 = conn2.cursor()
        
        # Writer (Manager)
        manager.record_ingestion_run("run_1", "source", 0)
        
        # Reader (External connection) - Should be able to read while WAL is active
        cur2.execute("SELECT id FROM ingestion_runs WHERE id='run_1'")
        row = cur2.fetchone()
        assert row is not None
        assert row[0] == "run_1"
        
        conn2.close()
        print("Verified: Concurrent read succeeded via WAL.")
        
    finally:
        manager.conn.close()
