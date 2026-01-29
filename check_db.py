import sqlite3
from pathlib import Path

db_path = Path(r'c:\Users\paolo\Desktop\cubo\storage\cubo_index\chroma.db')
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", [t[0] for t in tables])
    
    # Count entries in key tables
    for table in ['embeddings', 'documents', 'collections', 'seq']:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f"{table}: {count}")
        except Exception as e:
            print(f"{table}: error - {e}")
    
    # Check collections specifically
    try:
        cursor.execute('SELECT * FROM collections LIMIT 5')
        cols = cursor.fetchall()
        print(f"\nCollections sample:\n{cols}")
    except:
        pass
    
    conn.close()
else:
    print("Database file not found!")
