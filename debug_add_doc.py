from cubo.config import config
from cubo.retrieval.retriever import DocumentRetriever
from unittest.mock import MagicMock
import tempfile, sqlite3
from pathlib import Path

p = Path(tempfile.mkdtemp())
path = p / 'faiss'
path.mkdir()
config.set('vector_store_path', str(path))
config.set('vector_store_backend', 'faiss')

mock_model = MagicMock()
mock_model.get_sentence_embedding_dimension.return_value = 64

def mock_encode(texts, batch_size=1):
    return [[0.1]*64 for _ in texts]

mock_model.encode.side_effect = mock_encode

retr = DocumentRetriever(model=mock_model)
# build embeddings & insert using public API
retr.add_documents(['Test document 1'])
print('collection count (after add_documents):', retr.collection.count())
# Inspect DB
conn = sqlite3.connect(str(retr.collection._db_path))
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM documents')
print('DB count direct:', c.fetchone()[0])
conn.close()
