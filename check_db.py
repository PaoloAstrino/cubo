#!/usr/bin/env python3
import sqlite3
import os

db_path = 'evaluation/evaluation.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print('Tables:', tables)
    if 'evaluations' in tables:
        total = conn.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]
        unevaluated = conn.execute('SELECT COUNT(*) FROM evaluations WHERE answer_relevance_score IS NULL').fetchone()[0]
        print(f'Total queries: {total}')
        print(f'Unevaluated queries: {unevaluated}')
        recent = conn.execute('SELECT id, question, timestamp FROM evaluations ORDER BY timestamp DESC LIMIT 3').fetchall()
        print('Recent queries:')
        for row in recent:
            print(f'  ID {row[0]}: {row[1][:50]}... ({row[2]})')
    conn.close()
else:
    print('Database does not exist')