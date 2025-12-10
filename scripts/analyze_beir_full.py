"""
Analyze BEIR benchmark results (results/tonight_full/benchmark_beir_full.json)
Saves:
 - results/tonight_full/analysis_full.json
 - results/tonight_full/top_bottom_examples.json
 - results/tonight_full/analysis_full.csv
"""
import json, os, math, statistics, csv
from collections import Counter, defaultdict
import numpy as np

path = 'results/tonight_full/benchmark_beir_full.json'
if not os.path.exists(path):
    raise SystemExit(f"File not found: {path}")

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', [])
print('Loaded', len(results), 'results')

# Helper
def is_valid(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

# Collect arrays
latencies = [r.get('latency_ms', 0) for r in results]
faiths = [r.get('faithfulness') for r in results]
relevs = [r.get('relevancy') for r in results]
prec = [r.get('precision') for r in results]
recall = [r.get('recall') for r in results]

nan_counts = {
    'faithfulness': sum(1 for v in faiths if not is_valid(v)),
    'relevancy': sum(1 for v in relevs if not is_valid(v)),
    'precision': sum(1 for v in prec if not is_valid(v)),
    'recall': sum(1 for v in recall if not is_valid(v)),
}

# context metrics
context_counts = []
context_length_list = []
queries_with_no_contexts = 0

# similarity extraction (per-document sims or per-query sims)
per_query_avg_sim = []
per_doc_sims_count = 0

for r in results:
    # contexts
    contexts = r.get('contexts') or r.get('retrieved_contexts') or r.get('contexts_list') or r.get('context') or []
    if not contexts:
        queries_with_no_contexts += 1
        context_counts.append(0)
    else:
        if isinstance(contexts, list):
            context_counts.append(len(contexts))
            for c in contexts:
                if c and isinstance(c, str):
                    context_length_list.append(len(c))
        else:
            context_counts.append(1)
            if isinstance(contexts, str):
                context_length_list.append(len(contexts))
    # similarity check
    # some run results include "docs" or "retrieved_docs" as objects; check those
    sims = []
    if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict):
        for d in contexts:
            if 'similarity' in d and isinstance(d.get('similarity'), (int,float)):
                sims.append(d.get('similarity'))
    # fallback keys
    docs = r.get('docs') or r.get('retrieved_docs') or r.get('results')
    if isinstance(docs, list) and docs and isinstance(docs[0], dict) and 'similarity' in docs[0]:
        for d in docs:
            if isinstance(d.get('similarity'), (int,float)):
                sims.append(d.get('similarity'))
    if sims:
        per_doc_sims_count += len(sims)
        per_query_avg_sim.append(sum(sims)/len(sims))
    else:
        # try top similarity saved as 'similarity' on the result
        s = r.get('similarity') or r.get('top_similarity') or r.get('score') or None
        if isinstance(s, (int,float)):
            per_query_avg_sim.append(s)

# Stats
summary = {}
summary['total_queries'] = len(results)
summary['nan_counts'] = nan_counts
summary['latency'] = {
    'mean': statistics.mean(latencies) if latencies else 0,
    'median': statistics.median(latencies) if latencies else 0,
    'p90': np.percentile(latencies, 90) if latencies else 0,
    'max': max(latencies) if latencies else 0,
    'min': min(latencies) if latencies else 0,
}
summary['faith'] = {
    'mean': statistics.mean([v for v in faiths if is_valid(v)]) if any(is_valid(v) for v in faiths) else float('nan'),
    'median': statistics.median([v for v in faiths if is_valid(v)]) if any(is_valid(v) for v in faiths) else float('nan'),
    'p90': np.percentile([v for v in faiths if is_valid(v)], 90) if any(is_valid(v) for v in faiths) else float('nan'),
}
summary['relev'] = {
    'mean': statistics.mean([v for v in relevs if is_valid(v)]) if any(is_valid(v) for v in relevs) else float('nan'),
    'median': statistics.median([v for v in relevs if is_valid(v)]) if any(is_valid(v) for v in relevs) else float('nan'),
    'p90': np.percentile([v for v in relevs if is_valid(v)], 90) if any(is_valid(v) for v in relevs) else float('nan'),
}
summary['precision'] = {
    'mean': statistics.mean([v for v in prec if is_valid(v)]) if any(is_valid(v) for v in prec) else float('nan'),
}
summary['contexts'] = {
    'queries_with_no_contexts': queries_with_no_contexts,
    'avg_contexts_per_query': statistics.mean(context_counts) if context_counts else 0,
    'avg_context_len_chars': statistics.mean(context_length_list) if context_length_list else 0,
    'median_context_len_chars': statistics.median(context_length_list) if context_length_list else 0,
    'p90_context_len_chars': np.percentile(context_length_list,90) if context_length_list else 0,
}
summary['similarity'] = {
    'per_query_avg_count': len(per_query_avg_sim),
    'per_doc_sims_count': per_doc_sims_count,
    'avg_per_query_sim': statistics.mean(per_query_avg_sim) if per_query_avg_sim else None,
}

# Top/bottom queries
valid_faith = [(i,r) for i,r in enumerate(results) if is_valid(r.get('faithfulness'))]
valid_relev = [(i,r) for i,r in enumerate(results) if is_valid(r.get('relevancy'))]

sorted_by_faith = sorted(valid_faith, key=lambda x: x[1].get('faithfulness'))
sorted_by_relev = sorted(valid_relev, key=lambda x: x[1].get('relevancy'))

summary['bottom_5_lowest_faith'] = [{'index':i,'query':r.get('query'),'faith':r.get('faithfulness'),'relev':r.get('relevancy')} for i,r in sorted_by_faith[:5]]
summary['top_5_faith'] = [{'index':i,'query':r.get('query'),'faith':r.get('faithfulness'),'relev':r.get('relevancy')} for i,r in sorted_by_faith[-5:]]
summary['top_5_relev'] = [{'index':i,'query':r.get('query'),'relev':r.get('relevancy'),'faith':r.get('faithfulness')} for i,r in sorted_by_relev[-5:]]
summary['bottom_5_relev'] = [{'index':i,'query':r.get('query'),'relev':r.get('relevancy'),'faith':r.get('faithfulness')} for i,r in sorted_by_relev[:5]]

# Correlations
from scipy.stats import pearsonr
pairs = [(r.get('relevancy'), r.get('faithfulness'), r.get('latency_ms')) for r in results if is_valid(r.get('relevancy')) and is_valid(r.get('faithfulness'))]
if pairs:
    relev_arr = [p[0] for p in pairs]
    faith_arr = [p[1] for p in pairs]
    lat_arr = [p[2] for p in pairs]
    try:
        summary['correlation_relev_faith'] = pearsonr(relev_arr, faith_arr)[0]
    except Exception:
        summary['correlation_relev_faith'] = None
    try:
        summary['correlation_latency_faith'] = pearsonr(lat_arr, faith_arr)[0]
    except Exception:
        summary['correlation_latency_faith'] = None
else:
    summary['correlation_relev_faith'] = None
    summary['correlation_latency_faith'] = None

# Domain breakdown
domains = Counter(r.get('domain','unknown') for r in results)
summary['domain_counts'] = domains.most_common(20)

# Save small summary
out_dir = 'results/tonight_full'
with open(os.path.join(out_dir, 'analysis_full.json'), 'w', encoding='utf-8') as out:
    json.dump(summary, out, indent=2)

# Save per-query CSV for deeper inspection
csv_path = os.path.join(out_dir, 'analysis_full.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)
    header = ['index','query','answer','contexts_count','avg_context_len_chars','latency_ms','faithfulness','relevancy','precision','recall','domain','avg_similarity']
    writer.writerow(header)
    for i,r in enumerate(results):
        query = r.get('query')
        ans = r.get('answer')
        contexts = r.get('contexts') or r.get('retrieved_contexts') or []
        contexts_list = []
        if isinstance(contexts, list):
            contexts_list = contexts
        elif contexts:
            contexts_list = [contexts]
        avg_ctx_len = statistics.mean([len(c) for c in contexts_list]) if contexts_list else 0
        # similarity
        sims = []
        if isinstance(contexts_list, list) and contexts_list and isinstance(contexts_list[0], dict):
            for d in contexts_list:
                if 'similarity' in d and isinstance(d.get('similarity'), (int,float)):
                    sims.append(d.get('similarity'))
        if not sims:
            s = r.get('similarity') or r.get('top_similarity') or None
            if isinstance(s, (int,float)):
                sims.append(s)
        avg_sim = statistics.mean(sims) if sims else None
        writer.writerow([i, query, (ans[:200] + '...') if ans else '', len(contexts_list), int(avg_ctx_len), r.get('latency_ms'), r.get('faithfulness'), r.get('relevancy'), r.get('precision'), r.get('recall'), r.get('domain'), avg_sim])

# Save top/bottom examples for quick review
examples = {
    'worst_faith': [],
    'best_faith': [],
    'worst_relev': [],
    'best_relev': [],
}
for k,items in [('worst_faith', sorted_by_faith[:10]), ('best_faith', sorted_by_faith[-10:]), ('worst_relev', sorted_by_relev[:10]), ('best_relev', sorted_by_relev[-10:])]:
    for i,r in items:
        examples[k].append({'index':i,'query':r.get('query'),'answer':r.get('answer'),'contexts': r.get('contexts') or r.get('retrieved_contexts') or r.get('context') or [] ,'faith':r.get('faithfulness'),'relev':r.get('relevancy')})

with open(os.path.join(out_dir, 'top_bottom_examples.json'), 'w', encoding='utf-8') as exf:
    json.dump(examples, exf, indent=2)

print('Saved summary and csv and examples to results/tonight_full')
print('Summary preview:')
for k in ['total_queries','latency','faith','relev','contexts','similarity','nan_counts']:
    print(k, ':', summary.get(k))

print('\nDone')
