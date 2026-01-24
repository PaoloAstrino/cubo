import json, re, time
from cubo.core import CuboCore


def load_test_queries(path='data/beir/scifact/queries.jsonl', num=50):
    res=[]
    with open(path,'r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=num: break
            res.append(json.loads(line))
    return res


def find_sentence_with_keyword(text, keyword):
    parts = re.split(r'(?<=[.!?])\s+', text)
    for p in parts:
        if keyword.lower() in p.lower():
            return p.strip()
    return None


def evaluate_topk(cubo, queries, top_k, max_context_chars=4000, max_total_context_chars=12000):
    total=0
    doc_hit=0
    sent_hit=0
    for q in queries:
        meta=q.get('metadata',{})
        if not meta:
            continue
        total+=1
        gt_doc=list(meta.keys())[0]
        query_text=q['text']
        chunks=cubo.query_retrieve(query=query_text, top_k=top_k)
        retrieved_doc_ids=[c.get('doc_id') or c.get('id') for c in chunks[:top_k]]
        if str(gt_doc) in map(str,retrieved_doc_ids):
            doc_hit+=1
        # build contexts
        contexts=[]
        for chunk in chunks[:top_k]:
            text=chunk.get('text') or chunk.get('content') or chunk.get('document') or chunk.get('doc_text') or ''
            text=text.strip() if isinstance(text,str) else ''
            if text:
                if max_context_chars and len(text)>max_context_chars:
                    text=text[:max_context_chars]
                contexts.append(text)
        if max_total_context_chars:
            total_chars=0
            trunc=[]
            for t in contexts:
                if total_chars+len(t) <= max_total_context_chars:
                    trunc.append(t); total_chars+=len(t)
                else:
                    rem=max_total_context_chars-total_chars
                    if rem>0:
                        trunc.append(t[:rem]); total_chars+=rem
                    break
            contexts=trunc
        ctx_str='\n\n'.join(contexts)
        # load doc text
        doc_text=''
        with open('data/beir/scifact/corpus.jsonl','r',encoding='utf-8') as f:
            for line in f:
                d=json.loads(line)
                if d.get('_id')==gt_doc:
                    doc_text=d.get('text','')
                    break
        supp_sent=None
        if doc_text:
            for w in query_text.split():
                if len(w)>3:
                    s=find_sentence_with_keyword(doc_text,w)
                    if s:
                        supp_sent=s; break
        if not supp_sent and doc_text:
            supp_sent=doc_text[:200]
        if supp_sent and supp_sent.strip() in ctx_str:
            sent_hit+=1
    return total, doc_hit, sent_hit


if __name__=='__main__':
    cubo=CuboCore()
    print('Initializing CUBO components...')
    if not cubo.initialize_components():
        raise SystemExit('Failed to init')
    t0=time.time()
    while not (getattr(cubo,'retriever',None) and hasattr(cubo.retriever,'retrieve_top_documents')):
        if time.time()-t0>60: raise SystemExit('Timed out waiting for retriever')
        time.sleep(0.2)
    queries=load_test_queries()
    results={}
    for top_k in [3,5,10,20]:
        print(f'Evaluating top_k={top_k}...')
        total, doc_hit, sent_hit = evaluate_topk(cubo, queries, top_k)
        results[top_k]={'total': total, 'doc_in_topk': doc_hit, 'doc_recall': doc_hit/total if total else 0, 'sent_in_contexts': sent_hit, 'sent_recall': sent_hit/total if total else 0}
        print('  ', results[top_k])
    print('\nSummary:')
    for k,v in results.items():
        print(f' top_k={k}: doc_recall={v["doc_recall"]:.2f}, sent_recall={v["sent_recall"]:.2f}')
    # save results
    with open('results/rag_recall_grid_scifact.json','w',encoding='utf-8') as f:
        json.dump(results,f,indent=2)
    print('\nResults written to results/rag_recall_grid_scifact.json')