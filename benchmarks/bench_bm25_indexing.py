import time

from src.cubo.retrieval.bm25_store_factory import get_bm25_store

N = 5000
texts = [f"document {i} apples bananas cars and {i}" for i in range(N)]
docs = [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(texts)]

py = get_bm25_store("python")
start = time.time()
py.index_documents(docs)
end = time.time()
print("Python BM25 index time secs:", end - start)

try:
    whoosh = get_bm25_store("whoosh", index_dir="./whoosh_bench2")
    start = time.time()
    whoosh.index_documents(docs)
    end = time.time()
    print("Whoosh BM25 index time secs:", end - start)
except Exception as e:
    print("Whoosh benchmarking skipped:", e)
