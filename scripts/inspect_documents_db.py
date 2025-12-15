import sqlite3
p='results/tonight_full/storage/documents.db'
print('DB file:',p)
conn=sqlite3.connect(p)
c=conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('Tables:', c.fetchall())
try:
    c.execute('SELECT count(*) FROM documents')
    print('documents count:', c.fetchone()[0])
except Exception as e:
    print('Error counting documents:', e)
conn.close()