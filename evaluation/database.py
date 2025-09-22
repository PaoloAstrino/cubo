"""
CUBO Comprehensive Evaluation System
Stores detailed evaluation data and provides advanced analytics.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QueryEvaluation:
    """Complete evaluation data for a single query."""
    timestamp: str
    session_id: str
    question: str
    answer: str
    response_time: float  # seconds
    contexts: List[str]
    context_metadata: List[Dict[str, Any]]  # filename, chunk_id, similarity_score, etc.
    model_used: str
    embedding_model: str
    retrieval_method: str
    chunking_method: str

    # RAG Triad Metrics
    answer_relevance_score: float  # 0-1
    context_relevance_score: float  # 0-1
    groundedness_score: float  # 0-1

    # Detailed Metrics
    answer_length: int
    context_count: int
    total_context_length: int
    average_context_similarity: float
    answer_confidence: float  # if available

    # Quality Flags
    has_answer: bool
    is_fallback_response: bool
    error_occurred: bool
    error_message: Optional[str]

    # LLM-based Metrics
    llm_metrics: Optional[Dict[str, Any]] = None

    # User Feedback (future)
    user_rating: Optional[int] = None  # 1-5 stars
    user_feedback: Optional[str] = None

class EvaluationDatabase:
    """SQLite database for storing and analyzing evaluation data."""

    def __init__(self, db_path: str = "evaluation/evaluation.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._ensure_db_exists()
        self._create_tables()

    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _create_tables(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    response_time REAL,
                    contexts TEXT,  -- JSON array
                    context_metadata TEXT,  -- JSON array
                    model_used TEXT,
                    embedding_model TEXT,
                    retrieval_method TEXT,
                    chunking_method TEXT,

                    -- RAG Metrics
                    answer_relevance_score REAL,
                    context_relevance_score REAL,
                    groundedness_score REAL,

                    -- Detailed Metrics
                    answer_length INTEGER,
                    context_count INTEGER,
                    total_context_length INTEGER,
                    average_context_similarity REAL,
                    answer_confidence REAL,

                    -- Quality Flags
                    has_answer BOOLEAN,
                    is_fallback_response BOOLEAN,
                    error_occurred BOOLEAN,
                    error_message TEXT,

                    -- LLM Metrics
                    llm_metrics TEXT,

                    -- User Feedback
                    user_rating INTEGER,
                    user_feedback TEXT
                )
            ''')

            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON evaluations(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_answer_relevance ON evaluations(answer_relevance_score)')

    def store_evaluation(self, evaluation: QueryEvaluation):
        """Store a query evaluation in the database."""
        data = asdict(evaluation)

        # Convert complex types to JSON
        data['contexts'] = json.dumps(data['contexts'])
        data['context_metadata'] = json.dumps(data['context_metadata'])
        data['llm_metrics'] = json.dumps(data['llm_metrics'])

        with sqlite3.connect(self.db_path) as conn:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            values = list(data.values())

            conn.execute(f'''
                INSERT INTO evaluations ({columns})
                VALUES ({placeholders})
            ''', values)

        logger.info(f"Stored evaluation for query: {evaluation.question[:50]}...")

    def get_recent_evaluations(self, limit: int = 100) -> List[QueryEvaluation]:
        """Get recent evaluations."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM evaluations
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,)).fetchall()

        return [self._row_to_evaluation(row) for row in rows]

    def get_evaluations_by_date_range(self, start_date: str, end_date: str) -> List[QueryEvaluation]:
        """Get evaluations within date range."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM evaluations
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date)).fetchall()

        return [self._row_to_evaluation(row) for row in rows]

    def _row_to_evaluation(self, row) -> QueryEvaluation:
        """Convert database row to QueryEvaluation object."""
        data = dict(row)

        # Parse JSON fields
        data['contexts'] = json.loads(data['contexts'] or '[]')
        data['context_metadata'] = json.loads(data['context_metadata'] or '[]')
        data['llm_metrics'] = json.loads(data['llm_metrics'] or 'null')

        return QueryEvaluation(**data)

    def get_metrics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            total_queries = conn.execute('''
                SELECT COUNT(*) FROM evaluations
                WHERE timestamp >= ?
            ''', (cutoff_date,)).fetchone()[0]

            successful_queries = conn.execute('''
                SELECT COUNT(*) FROM evaluations
                WHERE timestamp >= ? AND error_occurred = 0
            ''', (cutoff_date,)).fetchone()[0]

            # Average scores
            scores = conn.execute('''
                SELECT
                    AVG(answer_relevance_score) as avg_answer_relevance,
                    AVG(context_relevance_score) as avg_context_relevance,
                    AVG(groundedness_score) as avg_groundedness,
                    AVG(response_time) as avg_response_time,
                    AVG(answer_length) as avg_answer_length
                FROM evaluations
                WHERE timestamp >= ? AND error_occurred = 0
            ''', (cutoff_date,)).fetchone()

            # Score distributions
            score_distributions = {}
            for metric in ['answer_relevance_score', 'context_relevance_score', 'groundedness_score']:
                dist = conn.execute(f'''
                    SELECT
                        COUNT(CASE WHEN {metric} >= 0.8 THEN 1 END) as excellent,
                        COUNT(CASE WHEN {metric} >= 0.6 AND {metric} < 0.8 THEN 1 END) as good,
                        COUNT(CASE WHEN {metric} >= 0.4 AND {metric} < 0.6 THEN 1 END) as fair,
                        COUNT(CASE WHEN {metric} < 0.4 THEN 1 END) as poor
                    FROM evaluations
                    WHERE timestamp >= ? AND error_occurred = 0 AND {metric} IS NOT NULL
                ''', (cutoff_date,)).fetchone()
                score_distributions[metric] = dict(dist)

            # Top performing queries
            top_queries = conn.execute('''
                SELECT question, answer_relevance_score, context_relevance_score, groundedness_score
                FROM evaluations
                WHERE timestamp >= ? AND error_occurred = 0
                ORDER BY (answer_relevance_score + context_relevance_score + groundedness_score) / 3 DESC
                LIMIT 5
            ''', (cutoff_date,)).fetchall()

            # Error analysis
            error_types = conn.execute('''
                SELECT error_message, COUNT(*) as count
                FROM evaluations
                WHERE timestamp >= ? AND error_occurred = 1
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
            ''', (cutoff_date,)).fetchall()

            return {
                'period_days': days,
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / max(total_queries, 1),
                'average_scores': dict(scores) if scores else {},
                'score_distributions': score_distributions,
                'top_performing_queries': [dict(row) for row in top_queries],
                'common_errors': [dict(row) for row in error_types],
                'generated_at': datetime.now().isoformat()
            }

    def export_to_csv(self, output_path: str, days: int = 30):
        """Export evaluation data to CSV."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM evaluations
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_date,))

        # Flatten JSON columns for CSV
        df['contexts'] = df['contexts'].apply(lambda x: json.loads(x) if x else [])
        df['context_metadata'] = df['context_metadata'].apply(lambda x: json.loads(x) if x else [])

        # Save to CSV (JSON columns will be stringified)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} evaluations to {output_path}")

    def get_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        with sqlite3.connect(self.db_path) as conn:
            # Daily averages
            daily_stats = pd.read_sql_query('''
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as query_count,
                    AVG(answer_relevance_score) as avg_answer_relevance,
                    AVG(context_relevance_score) as avg_context_relevance,
                    AVG(groundedness_score) as avg_groundedness,
                    AVG(response_time) as avg_response_time,
                    SUM(error_occurred) as error_count
                FROM evaluations
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', conn, params=((datetime.now() - timedelta(days=days)).isoformat(),))

        if daily_stats.empty:
            return {'error': 'No data available for trend analysis'}

        # Calculate trends
        trends = {}
        for metric in ['avg_answer_relevance', 'avg_context_relevance', 'avg_groundedness', 'avg_response_time']:
            if len(daily_stats) > 1:
                values = daily_stats[metric].dropna()
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    trends[metric] = {
                        'slope': trend,
                        'direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
                        'change_per_day': trend
                    }

        return {
            'daily_stats': daily_stats.to_dict('records'),
            'trends': trends,
            'summary': {
                'total_days': len(daily_stats),
                'avg_daily_queries': daily_stats['query_count'].mean(),
                'best_day': daily_stats.loc[daily_stats['avg_answer_relevance'].idxmax()]['date'] if not daily_stats.empty else None,
                'worst_day': daily_stats.loc[daily_stats['avg_answer_relevance'].idxmin()]['date'] if not daily_stats.empty else None
            }
        }