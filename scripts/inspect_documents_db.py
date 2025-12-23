import os
import sqlite3

p = "results/tonight_full/storage/documents.db"
print("DB:", os.path.abspath(p))
conn = sqlite3.connect(p)
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in c.fetchall()]
print("Tables:", tables)

for t in tables:
    print("\n--- TABLE", t)
    try:
        schema = list(c.execute(f"PRAGMA table_info('{t}')"))
        print("schema:", schema)
    except Exception as e:
        print("schema error", e)
    try:
        cnt = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print("count:", cnt)
    except Exception as e:
        print("count error", e)

    if t == "documents":
        rows = c.execute(
            "SELECT id, substr(content,1,200) as snippet, length(content) as len FROM documents ORDER BY rowid LIMIT 5"
        ).fetchall()
        for r in rows:
            print("\nid:", r[0])
            print("len:", r[2])
            print("snippet:", r[1].replace("\n", " ").strip())

# Additional stats
try:
    min_len, avg_len, max_len = c.execute(
        "SELECT MIN(LENGTH(content)), AVG(LENGTH(content)), MAX(LENGTH(content)) FROM documents"
    ).fetchone()
    print("\nDocument length stats (min, avg, max):", min_len, avg_len, max_len)
except Exception as e:
    print("len stats error", e)

conn.close()
