import json
import sqlite3

import pandas as pd


def get_stats():
    try:
        conn = sqlite3.connect("data/metadata.db")
        df = pd.read_sql_query(
            """
            SELECT id, chunks_count, started_at, finished_at 
            FROM ingestion_runs 
            WHERE status = 'completed' AND finished_at IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT 5
        """,
            conn,
        )
        conn.close()

        if df.empty:
            print("[]")
            return

        results = []
        for _, row in df.iterrows():
            try:
                # Handle potentially different timestamp formats
                start = pd.to_datetime(row["started_at"])
                end = pd.to_datetime(row["finished_at"])

                duration = (end - start).total_seconds()
                if duration > 0:
                    speed = row["chunks_count"] / duration
                    results.append(
                        {
                            "id": row["id"],
                            "chunks": row["chunks_count"],
                            "duration_sec": round(duration, 2),
                            "speed_chunks_per_sec": round(speed, 2),
                        }
                    )
            except Exception:
                continue

        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_stats()
