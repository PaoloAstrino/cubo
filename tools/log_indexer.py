#!/usr/bin/env python3
"""
Incremental log indexer: index JSONL logs into a SQLite FTS5 database for offline querying.
"""
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path


def ensure_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Create FTS table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS log_meta (id INTEGER PRIMARY KEY, key TEXT UNIQUE, value TEXT)"""
    )
    cur.execute(
        """CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(message, level, component, timestamp, trace_id, context, logfile);"""
    )
    conn.commit()
    return conn


def get_offset(conn, logfile: str):
    cur = conn.cursor()
    cur.execute("SELECT value FROM log_meta WHERE key = ?", (f"offset:{logfile}",))
    row = cur.fetchone()
    if not row:
        return 0
    return int(row[0])


def set_offset(conn, logfile: str, value: int):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO log_meta (key, value) VALUES (?, ?)",
        (f"offset:{logfile}", str(value)),
    )
    conn.commit()


def index_file(logfile: str, db_path: str):
    dbp = Path(db_path)
    dbp.parent.mkdir(parents=True, exist_ok=True)
    conn = ensure_db(dbp)
    offset = get_offset(conn, logfile)
    with open(logfile, encoding="utf-8") as f:
        f.seek(offset)
        cur = conn.cursor()
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # Not JSON — index as message text
                rec = {
                    "message": line,
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "INFO",
                    "component": "log",
                }
            message = rec.get("message") or rec.get("msg") or ""
            level = rec.get("level", "")
            component = rec.get("component", "") or rec.get("name", "")
            timestamp = rec.get("asctime", rec.get("timestamp", datetime.utcnow().isoformat()))
            trace_id = rec.get("trace_id", "")
            context = json.dumps(rec.get("context", {})) if rec.get("context") else ""
            # Insert into FTS table
            cur.execute(
                "INSERT INTO logs_fts (message, level, component, timestamp, trace_id, context, logfile) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message, level, component, timestamp, trace_id, context, logfile),
            )
        # Record new offset
        new_offset = f.tell()
        set_offset(conn, logfile, new_offset)
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Index JSONL logs into SQLite FTS")
    parser.add_argument("--log-file", required=True, help="Path to JSONL log file")
    parser.add_argument(
        "--db", required=False, default="./logs/index/logs.db", help="SQLite DB path for index"
    )
    args = parser.parse_args()
    index_file(args.log_file, args.db)


if __name__ == "__main__":
    main()
