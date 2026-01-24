import time
import json
from cubo.core import CuboCore


def load_queries_with_metadata(path='data/beir/scifact/queries.jsonl', limit=20):
    res = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d.get('metadata'):
                res.append(d)
                if len(res) >= limit:
                    break
    return res


if __name__ == '__main__':
    cubo = CuboCore()
    print('Initializing CUBO components...')
    if not cubo.initialize_components():
        raise SystemExit('Failed to initialize components')
    # wait for retriever
    t0 = time.time()
    while not (getattr(cubo, 'retriever', None) and hasattr(cubo.retriever, 'retrieve_top_documents')):
        if time.time() - t0 > 60:
            raise SystemExit('Timed out waiting for retriever')
        time.sleep(0.5)

    queries = load_queries_with_metadata()
    for q in queries:
        qid = q.get('_id')
        text = q.get('text')
        meta = q.get('metadata')
        gt_doc_ids = list(meta.keys())
        # retrieve
        chunks = cubo.query_retrieve(query=text, top_k=3)
        retrieved_doc_ids = [c.get('doc_id') or c.get('id') or c.get('metadata', {}).get('doc_id') for c in chunks[:3]]
        print('Query', qid)
        print('  text:', text[:120])
        print('  gt_doc_ids:', gt_doc_ids)
        print('  retrieved_doc_ids:', retrieved_doc_ids)
        hit = any(str(d) in map(str, retrieved_doc_ids) for d in gt_doc_ids)
        print('  ground-truth in top-3?', hit)
        print('')
