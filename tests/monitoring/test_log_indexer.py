import json
import sqlite3

from scripts.log_indexer import index_file


def test_indexer_basic(tmp_path):
    # Prepare a small JSONL log file
    logf = tmp_path / "test_log.jsonl"
    entries = [
        {
            "message": "Error processing file",
            "level": "ERROR",
            "component": "ingest",
            "timestamp": "2025-11-21T00:00:00Z",
            "trace_id": "t1",
        },
        {
            "message": "Search completed",
            "level": "INFO",
            "component": "retriever",
            "timestamp": "2025-11-21T00:01:00Z",
            "trace_id": "t2",
        },
    ]
    with open(logf, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    db_path = tmp_path / "logs.db"
    index_file(str(logf), str(db_path))

    # Validate index content directly with SQLite
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM logs_fts")
    count = cur.fetchone()[0]
    assert count >= 2
    # Query by trace_id
    cur.execute("SELECT message FROM logs_fts WHERE trace_id = ?", ("t1",))
    rows = cur.fetchall()
    assert rows and rows[0][0] == "Error processing file"
    conn.close()
