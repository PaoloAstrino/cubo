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


def _calculate_recall(hit_ids: List[str], relevant_ids: set) -> float:
    """Calculate recall for a single query."""
    num_relevant_retrieved = len(relevant_ids.intersection(set(hit_ids)))
    return num_relevant_retrieved / len(relevant_ids) if len(relevant_ids) > 0 else 0


def _calculate_mrr(hit_ids: List[str], relevant_ids: set) -> float:
    """Calculate MRR (Mean Reciprocal Rank) for a single query."""
    for i, hid in enumerate(hit_ids):
        if hid in relevant_ids:
            return 1 / (i + 1)
    return 0


def _calculate_dcg(hit_ids: List[str], relevant_ids: set, qdata) -> float:
    """Calculate DCG (Discounted Cumulative Gain)."""
    dcg = 0.0
    for i, hid in enumerate(hit_ids):
        if hid in relevant_ids:
            rel = 1.0  # binary
            if isinstance(qdata, dict):
                rel = float(qdata.get(hid, 1))  # use score if available
            dcg += rel / (import_math_log2(i + 2))
    return dcg


def _calculate_idcg(relevant_ids: set, qdata, k: int) -> float:
    """Calculate IDCG (Ideal DCG)."""
    sorted_rels = sorted(
        [float(qdata[rid]) if isinstance(qdata, dict) else 1.0 for rid in relevant_ids],
        reverse=True
    )[:k]
    
    idcg = 0.0
    for i, rel in enumerate(sorted_rels):
        idcg += rel / (import_math_log2(i + 2))
    return idcg


def _calculate_ndcg(hit_ids: List[str], relevant_ids: set, qdata, k: int) -> float:
    """Calculate NDCG (Normalized DCG)."""
    dcg = _calculate_dcg(hit_ids, relevant_ids, qdata)
    idcg = _calculate_idcg(relevant_ids, qdata, k)
    return dcg / idcg if idcg > 0 else 0


def _process_query_metrics(qid: str, results: Dict, qrels: Dict, k_values: List[int]) -> Dict:
    """Calculate all metrics for a single query across different k values."""
    qdata = qrels[qid]
    relevant_ids = set(str(k) for k in qdata.keys()) if isinstance(qdata, dict) else set(str(x) for x in qdata)
    
    all_hits = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
    query_metrics = {}
    
    for k in k_values:
        top_k_hits = all_hits[:k]
        hit_ids = [str(hid) for hid, _ in top_k_hits]
        
        query_metrics[k] = {
            "recall": _calculate_recall(hit_ids, relevant_ids),
            "mrr": _calculate_mrr(hit_ids, relevant_ids),
            "ndcg": _calculate_ndcg(hit_ids, relevant_ids, qdata, k)
        }
    
    return query_metrics


def compute_metrics(run_path, qrels_path, k_values=[10, 50, 100]):
    """Compute retrieval metrics from run results and qrels."""
    print(f"Loading run results from {run_path}...")
    with open(run_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loading qrels from {qrels_path}...")
    qrels = load_qrels(qrels_path)

    results = {qid: hits for qid, hits in data.items() if not qid.startswith('_')}
    
    matched = 0
    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for qid, qdata in qrels.items():
        if qid not in results:
            continue
        
        matched += 1
        query_metrics = _process_query_metrics(qid, results, qrels, k_values)
        
        for k in k_values:
            metrics[k]["recall"].append(query_metrics[k]["recall"])
            metrics[k]["mrr"].append(query_metrics[k]["mrr"])
            metrics[k]["ndcg"].append(query_metrics[k]["ndcg"])

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
