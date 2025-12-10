import csv, os, math, statistics, json
from collections import defaultdict
os.chdir('c:/Users/paolo/Desktop/cubo')

path='results/tonight_full/analysis_full.csv'
rows=list(csv.DictReader(open(path,'r',encoding='utf-8')))

# parse helper
def parse_float(v):
    if v in [None,'','NULL','None']:
        return None
    try:
        f=float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None

faith_vals=[parse_float(r['faithfulness']) for r in rows]
relev_vals=[parse_float(r['relevancy']) for r in rows]
lat_vals=[parse_float(r['latency_ms']) for r in rows]

# bins
def bucket_counts(values, bins=[0,0.25,0.5,0.75,1.0]):
    bcounts=defaultdict(int)
    for v in values:
        if v is None:
            bcounts['none']+=1
        else:
            placed=False
            for i in range(len(bins)-1):
                if v >= bins[i] and v < bins[i+1]:
                    bcounts[f'{bins[i]}-{bins[i+1]}']+=1
                    placed=True
                    break
            if not placed:
                if v==1.0:
                    bcounts['1.0']+=1
                else:
                    bcounts['>1?']+=1
    return bcounts

faith_buckets=bucket_counts(faith_vals)
relev_buckets=bucket_counts(relev_vals)

# find queries with extreme context length
context_field='contexts'
large_context_idxs=[]
for r in rows:
    contexts=r.get('contexts')
    # contexts string may be a long string representation of list; treat raw CSV limited
    if contexts:
        if len(contexts)>100000:
            large_context_idxs.append(int(r['index']))

# collect top/bottom indices from CSV
valid_faith = [(i,parse_float(r['faithfulness'])) for i,r in enumerate(rows) if parse_float(r['faithfulness']) is not None]
valid_relev = [(i,parse_float(r['relevancy'])) for i,r in enumerate(rows) if parse_float(r['relevancy']) is not None]

valid_faith_sorted=sorted(valid_faith, key=lambda x: x[1])
valid_relev_sorted=sorted(valid_relev, key=lambda x: x[1])

# pick extremes
worst_faith_idx=[i for i,_ in valid_faith_sorted[:10]]
best_faith_idx=[i for i,_ in valid_faith_sorted[-10:]]
worst_relev_idx=[i for i,_ in valid_relev_sorted[:10]]
best_relev_idx=[i for i,_ in valid_relev_sorted[-10:]]

# Collect sample rows
out = {
    'counts':{
        'faith_na': sum(1 for v in faith_vals if v is None),
        'relev_na': sum(1 for v in relev_vals if v is None),
    },
    'faith_buckets': faith_buckets,
    'relev_buckets': relev_buckets,
    'largest_context_indices': large_context_idxs[:20],
    'worst_faith': [],
    'best_faith': [],
    'worst_relev': [],
    'best_relev': []
}

for idx in worst_faith_idx:
    r=rows[idx]
    out['worst_faith'].append({'index':r['index'], 'query':r['query'],'faith':r['faithfulness'],'relev':r['relevancy'],'ans':r['answer'][:400]})
for idx in best_faith_idx:
    r=rows[idx]
    out['best_faith'].append({'index':r['index'], 'query':r['query'],'faith':r['faithfulness'],'relev':r['relevancy'],'ans':r['answer'][:400]})
for idx in worst_relev_idx:
    r=rows[idx]
    out['worst_relev'].append({'index':r['index'], 'query':r['query'],'relev':r['relevancy'],'faith':r['faithfulness'],'ans':r['answer'][:400]})
for idx in best_relev_idx:
    r=rows[idx]
    out['best_relev'].append({'index':r['index'], 'query':r['query'],'relev':r['relevancy'],'faith':r['faithfulness'],'ans':r['answer'][:400]})

# Save
open('results/tonight_full/analysis_more.json','w',encoding='utf-8').write(json.dumps(out,indent=2))
print('Saved results/tonight_full/analysis_more.json')
print('Counts:', out['counts'])
print('Faith buckets sample:', out['faith_buckets'])
print('Relev buckets sample:', out['relev_buckets'])
print('Large context count:', len(out['largest_context_indices']))
print('Sample worst faith index list:', [e['index'] for e in out['worst_faith']])
