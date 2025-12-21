
import json

with open('results/beir_run_optimized.json', 'r') as f:
    results = json.load(f)

qrels = {}
with open('data/beir/qrels/dev.tsv', 'r') as f:
    next(f)
    for line in f:
        qid, did, score = line.strip().split('\t')
        if qid not in qrels:
            qrels[qid] = set()
        qrels[qid].add(did)

found_hits = []
for qid, rels in qrels.items():
    if qid in results:
        intersection = rels & set(results[qid].keys())
        if intersection:
            found_hits.append((qid, intersection))

print(f"Total queries in results that are in qrels: {len([qid for qid in qrels if qid in results])}")
print(f"Queries with at least one hit in top 100: {len(found_hits)}")
if found_hits:
    print(f"Sample hit: Query {found_hits[0][0]} has hits {found_hits[0][1]}")
else:
    print("NO HITS FOUND IN TOP 100 FOR ANY QUERY")
