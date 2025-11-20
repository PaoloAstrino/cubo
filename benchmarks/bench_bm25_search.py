import time
from src.cubo.retrieval.bm25_store_factory import get_bm25_store
from src.cubo.retrieval.bm25_python_store import BM25PythonStore

# Prepare a sample dataset
N = 1000
texts = [f"document {i} apples bananas cars and {i}" for i in range(N)]
docs = [{'doc_id': f'd{i}', 'text': t} for i, t in enumerate(texts)]

# Test python store
py = get_bm25_store('python')
py.index_documents(docs)
q = 'apples'
start = time.time()
for i in range(100):
    py.search(q, top_k=10)
end = time.time()
print('Python BM25 avg ms:', (end - start) / 100 * 1000)

# Test whoosh store if available
whoosh = get_bm25_store('whoosh', index_dir='./whoosh_bench')
if whoosh:
    whoosh.index_documents(docs)
    start = time.time()
    for i in range(100):
        whoosh.search(q, top_k=10)
    end = time.time()
    print('Whoosh BM25 avg ms:', (end - start) / 100 * 1000)
