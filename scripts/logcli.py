#!/usr/bin/env python3
"""
CLI to search indexed logs stored in SQLite FTS database.
"""
import argparse
import sqlite3
from pathlib import Path


def query_index(
    db_path: Path, query: str, level: str = None, component: str = None, limit: int = 50
):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    sql = "SELECT rowid, message, level, component, timestamp, trace_id, context FROM logs_fts WHERE logs_fts MATCH ?"
    params = [query]
    if level:
        sql += " AND level = ?"
        params.append(level)
    if component:
        sql += " AND component = ?"
        params.append(component)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    for r in rows:
        print(f"[{r[4]}] {r[2]} {r[3]} trace={r[5]}: {r[1]}\n")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Query indexed logs")
    parser.add_argument("--db", default="./logs/index/logs.db", help="Path to index DB")
    parser.add_argument("--query", required=True, help="FTS query string")
    parser.add_argument("--level", required=False, help="Filter by level")
    parser.add_argument("--component", required=False, help="Filter by component")
    parser.add_argument("--limit", required=False, type=int, default=50, help="Result limit")
    args = parser.parse_args()
    query_index(
        Path(args.db), args.query, level=args.level, component=args.component, limit=args.limit
    )


if __name__ == "__main__":
    main()
