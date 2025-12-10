import json, os
from collections import defaultdict

os.chdir('c:/Users/paolo/Desktop/cubo')
data=json.load(open('results/tonight_full/benchmark_beir_full.json'))['results']
count=defaultdict(int)

for r in data:
    contexts=r.get('contexts') or r.get('retrieved_contexts') or []
    first=(contexts[0] if isinstance(contexts,list) and contexts else None)
    s=''
    if isinstance(first,dict):
        s=first.get('document') or first.get('text') or first.get('content') or ''
    elif isinstance(first,str):
        s=first
    else:
        s=str(first)
    count[s[:120]] += 1

counts = sorted(count.items(), key=lambda x: -x[1])
print('Top first-context snippets and counts (top10):')
for k,c in counts[:10]:
    print(c, k)

# Print some indices for the top frequent snippet
most=counts[0][0]
ids=[]
for i,r in enumerate(data):
    contexts=r.get('contexts') or r.get('retrieved_contexts') or []
    first=(contexts[0] if isinstance(contexts,list) and contexts else None)
    s='' 
    if isinstance(first,dict):
        s=first.get('document') or first.get('text') or first.get('content') or ''
    elif isinstance(first,str):
        s=first
    if s[:120]==most:
        ids.append(i)

print('Indices of top snippet (first 20):', ids[:20])
