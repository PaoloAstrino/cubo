#!/usr/bin/env python3
import sqlite3
import os

db_path = 'evaluation/evaluation.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    # Get the most recent query with all details
    recent = conn.execute('''
        SELECT id, question, answer, answer_relevance_score, context_relevance_score, groundedness_score, timestamp
        FROM evaluations
        ORDER BY timestamp DESC
        LIMIT 1
    ''').fetchone()
    if recent:
        print(f'Most recent query (ID {recent[0]}):')
        print(f'Question: {recent[1]}')
        print(f'Answer: {recent[2][:100]}...' if recent[2] else 'No answer')
        print(f'Answer Relevance: {recent[3]}')
        print(f'Context Relevance: {recent[4]}')
        print(f'Groundedness: {recent[5]}')
        print(f'Timestamp: {recent[6]}')
    conn.close()
else:
    print('Database does not exist')
