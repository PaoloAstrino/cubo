import json
from pathlib import Path
res_path = Path('results') / 'full_beir_retrieval_results.json'
if not res_path.exists():
    print('results file not found:', res_path)
    raise SystemExit(1)
res = json.load(open(res_path, 'r', encoding='utf-8'))
gt = json.load(open('data/beir/ground_truth.json', 'r', encoding='utf-8'))
klist = [5,10,20]
counts = {k:0 for k in klist}
questions = 0
for r in res['results'].get('easy', []):
    qid = r.get('question_id')
    if not qid:
        continue
    if qid not in gt:
        continue
    questions += 1
    retrieved = r.get('retrieved_ids', [])
    for k in klist:
        topk = set(retrieved[:k])
        if set(gt[qid]) & topk:
            counts[k] += 1
print('Total ground-truth-covered questions considered:', questions)
for k in klist:
    denom = questions if questions else 1
    print(f'top{k}: {counts[k]} => {counts[k]/denom:.4f}')
