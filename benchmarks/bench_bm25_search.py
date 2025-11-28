import time

from src.cubo.retrieval.bm25_store_factory import get_bm25_store

# Prepare a sample dataset
N = 1000
texts = [f"document {i} apples bananas cars and {i}" for i in range(N)]
docs = [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(texts)]

# Test python store
py = get_bm25_store("python")
py.index_documents(docs)
q = "apples"
start = time.time()
for i in range(100):
    py.search(q, top_k=10)
end = time.time()
print("Python BM25 avg ms:", (end - start) / 100 * 1000)

# Additional alternative backends can be benchmarked by replacing the backend
# name above. Whoosh backend has been removed; this benchmark focuses on the
# Python BM25 implementation.
