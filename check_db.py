#!/usr/bin/env python3
"""
Database Status Checker

This script checks the status of the evaluation database, including:
- Available tables
- Total number of queries
- Number of unevaluated queries
- Recent query history

Usage: python check_db.py
"""

import sqlite3
import os

# Database path
db_path = 'evaluation/evaluation.db'

if os.path.exists(db_path):
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Get list of all tables
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print('Tables:', tables)

    if 'evaluations' in tables:
        # Count total queries
        total = conn.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]

        # Count unevaluated queries (missing answer_relevance_score)
        unevaluated = conn.execute('SELECT COUNT(*) FROM evaluations WHERE answer_relevance_score IS NULL').fetchone()[0]

        print(f'Total queries: {total}')
        print(f'Unevaluated queries: {unevaluated}')

        # Get recent queries (last 3)
        recent = conn.execute('SELECT id, question, timestamp FROM evaluations ORDER BY timestamp DESC LIMIT 3').fetchall()
        print('Recent queries:')
        for row in recent:
            print(f'  ID {row[0]}: {row[1][:50]}... ({row[2]})')

    # Close database connection
    conn.close()
else:
    print('Database does not exist')
