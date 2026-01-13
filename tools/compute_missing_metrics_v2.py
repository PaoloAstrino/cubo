#!/usr/bin/env python3
import glob, os, re, subprocess, json

runs = sorted([r for r in glob.glob('results/beir_run_*.json') if not r.endswith('_metrics_k10.json')])
if not runs:
    print('No run files found.')
    exit(0)

# helper to find qrels
all_qrels = glob.glob('data/beir/**/qrels/test.tsv', recursive=True)

for r in runs:
    bn = os.path.basename(r)
    # extract dataset
    m = re.match(r'beir_run_(.+?)(?:_topk|\.json)', bn)
    if not m:
        print(f'WARNING: cannot parse dataset from {bn}; skipping')
        continue
    dataset = m.group(1)
    # possible dataset name variants to try
    candidates = [f'data/beir/{dataset}/qrels/test.tsv', f'data/beir/{dataset.replace("-","_")}/qrels/test.tsv', f'data/beir/{dataset.replace("_","-")}/qrels/test.tsv']
    qrels = None
    for c in candidates:
        if os.path.exists(c):
            qrels = c
            break
    if not qrels:
        # search in all_qrels for one that contains dataset substring
        for q in all_qrels:
            if dataset in q:
                qrels = q
                break
    if not qrels:
        print(f'WARNING: qrels missing for dataset {dataset}; skipping {bn}')
        continue
    metrics_fn = r.replace('.json','_metrics_k10.json')
    if os.path.exists(metrics_fn):
        print(f'SKIP (exists): {metrics_fn}')
        continue
    print(f'Computing metrics for {bn} using qrels {qrels}...')
    try:
        subprocess.run(['python','tools/calculate_beir_metrics.py','--results',r,'--qrels',qrels,'--k','10'], check=True)
        with open(metrics_fn,'r',encoding='utf-8') as mf:
            mdata = json.load(mf)
        print(json.dumps({'dataset':dataset,'run':bn,'metrics':mdata}))
    except subprocess.CalledProcessError as e:
        print(f'ERROR computing metrics for {r}: {e}')

print('Done.')
