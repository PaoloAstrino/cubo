import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from cubo.main import CUBOApp
import time, json

app = CUBOApp()
print('initialize_components:', app.initialize_components())
ret = app.retriever
print('retriever inited:', bool(ret))
count = 0
try:
    count = getattr(ret.collection, 'count', lambda: 0)()
except Exception as e:
    pass
print('collection_count:', count)
print('bm25 docs loaded:', len(getattr(ret.bm25, 'docs', [])))
queries = []
queries_path = os.path.join(project_root, 'data', 'beir', 'queries.jsonl')
if os.path.exists(queries_path):
    with open(queries_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            obj = json.loads(line.strip())
            queries.append(obj.get('text'))
else:
    # Fallback: use basic test question
    queries = ['What is the story about?']
for qi in queries:
    print('\nQuery:', qi)
    start = time.time(); emb = ret.executor.generate_query_embedding(qi); t1 = time.time();
    print('embedding_time:', t1 - start)
    start = time.time(); dense = ret.executor.query_dense(emb, 10, qi, ret.current_documents); t2 = time.time();
    print('dense_time:', t2 - start, 'dense_len:', len(dense))
    start = time.time(); bm = ret.executor.query_bm25(qi, 10, ret.current_documents); t3 = time.time();
    print('bm25_time:', t3 - start, 'bm25_len:', len(bm))
    start = time.time(); res = ret.retrieve_top_documents(qi, top_k=5); t4 = time.time();
    print('retrieve_top_time:', t4 - start, 'results:', len(res))
