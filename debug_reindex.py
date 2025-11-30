import runpy, sys, sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch
from cubo.config import config
import tempfile
from cubo.ingestion.deep_ingestor import DeepIngestor

p = Path(tempfile.mkdtemp())
path = p / 'faiss'
path.mkdir()
config.set('vector_store_path', str(path))
config.set('vector_store_backend', 'faiss')

# create parquet
folder = p / 'docs'
folder.mkdir()
(folder / 'a.txt').write_text('Test doc for reindex')
out = p / 'out'
ing = DeepIngestor(input_folder=str(folder), output_dir=str(out))
res = ing.ingest()
parquet = res['chunks_parquet']

with patch('src.cubo.embeddings.model_loader.model_manager.get_model') as mock_get_model:
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 64
    def mock_encode(texts, batch_size=1):
        return [[0.1]*64 for _ in texts]
    mock_model.encode.side_effect = mock_encode
    mock_get_model.return_value = mock_model
    script = str(Path.cwd() / 'scripts' / 'reindex_parquet.py')
    old_argv = sys.argv
    try:
        sys.argv = ['reindex_parquet.py', '--parquet', parquet, '--collection', 'test_reindex', '--replace-collection', '--wipe-db']
        runpy.run_path(script, run_name='__main__')
    finally:
        sys.argv = old_argv

from cubo.retrieval.retriever import DocumentRetriever
retr = DocumentRetriever(model=None)
print('retrieval db path:', retr.collection._db_path)
conn = sqlite3.connect(str(retr.collection._db_path))
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM documents')
print('db count via sqlite3:', c.fetchone()[0])
try:
    c.execute("PRAGMA table_info(documents)")
    print('columns:', c.fetchall())
    c.execute('SELECT id, content FROM documents')
    print('rows:', c.fetchall())
except Exception as e:
    print('error querying db:', e)
conn.close()

print('Index files:', list(path.iterdir()))
print('Tmpdir:', p)

print('Done')
