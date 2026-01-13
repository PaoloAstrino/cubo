import json
import argparse


def load_qrels(qrels_path):
    qrels = {}
    if qrels_path.endswith('.tsv'):
        with open(qrels_path, 'r', encoding='utf-8') as f:
            next(f, None)  # skip header if present
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    qrels.setdefault(qid, {})[did] = score
    else:
        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels = json.load(f)
    return qrels


def compute_metrics(run_path, qrels_path, k_values=[10, 50, 100]):
    print(f"Loading run results from {run_path}...")
    with open(run_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading qrels from {qrels_path}...")
    qrels = load_qrels(qrels_path)

    results = {qid: hits for qid, hits in data.items() if not qid.startswith('_')}
    
    matched = 0
    # Store aggregate metrics
    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for qid, qdata in qrels.items():
        if qid not in results:
            continue
        matched += 1
        relevant_ids = set(str(k) for k in qdata.keys()) if isinstance(qdata, dict) else set(str(x) for x in qdata)
        
        # Sort full hits once
        # Assuming we retrieved enough; if top-k in file is less than max(k_values), we handle it
        all_hits = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        
        for k in k_values:
            top_k_hits = all_hits[:k]
            hit_ids = [str(hid) for hid, _ in top_k_hits]

            # Recall
            num_relevant_retrieved = len(relevant_ids.intersection(set(hit_ids)))
            rec = num_relevant_retrieved / len(relevant_ids) if len(relevant_ids) > 0 else 0
            metrics[k]["recall"].append(rec)

            # MRR (simple implementation)
            mrr_val = 0
            for i, hid in enumerate(hit_ids):
                if hid in relevant_ids:
                    mrr_val = 1 / (i + 1)
                    break
            metrics[k]["mrr"].append(mrr_val)
            
            # NDCG (simple binary relevance for now, or use score if dict)
            # Skipping complex NDCG for brevity unless strictly needed, sticking to Recall/MRR as requested primarily
            # But user asked for NDCG@10. Let's add basic Binary NDCG
            dcg = 0.0
            idcg = 0.0
            for i, hid in enumerate(hit_ids):
                if hid in relevant_ids:
                    rel = 1.0 # binary
                    if isinstance(qdata, dict):
                        rel = float(qdata.get(hid, 1)) # use score if avail
                    dcg += rel / (import_math_log2(i + 2))
            
            # IDCG
            sorted_rels = sorted([float(qdata[rid]) if isinstance(qdata, dict) else 1.0 for rid in relevant_ids], reverse=True)[:k]
            for i, rel in enumerate(sorted_rels):
                idcg += rel / (import_math_log2(i + 2))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[k]["ndcg"].append(ndcg)

    if matched == 0:
        print("No matching queries found between run and qrels.")
        return None

    print(f"Queries matched: {matched}")
    print("-" * 40)
    for k in k_values:
        avg_rec = sum(metrics[k]["recall"]) / matched
        avg_mrr = sum(metrics[k]["mrr"]) / matched
        avg_ndcg = sum(metrics[k]["ndcg"]) / matched
        print(f"k={k:<3} | Recall: {avg_rec:.4f} | MRR: {avg_mrr:.4f} | NDCG: {avg_ndcg:.4f}")

    return metrics

import math
def import_math_log2(x):
    return math.log2(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--qrels', required=True)
    # k argument ignored now, using standard set
    args = parser.parse_args()
    compute_metrics(args.run, args.qrels)
