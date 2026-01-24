"""
BEIR Metrics Calculator

This script calculates standard information retrieval metrics (Recall@k, MRR, nDCG@k)
from BEIR benchmark results. It supports both TSV and JSON qrels formats.

Metrics calculated:
- Recall@k: Fraction of relevant documents retrieved in top-k
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant document
- nDCG@k: Normalized Discounted Cumulative Gain at rank k
"""

import json


def _load_results(results_path: str) -> dict:
    """Load and filter results from JSON file."""
    with open(results_path, "r") as f:
        data = json.load(f)
    # Filter out metadata fields (those starting with underscore)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _load_qrels(qrels_path: str) -> dict:
    """Load qrels from TSV or JSON format."""
    if qrels_path.endswith(".tsv"):
        qrels = {}
        with open(qrels_path, "r") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, did, score = parts[0], parts[1], int(parts[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = score
        return qrels
    else:
        with open(qrels_path, "r") as f:
            return json.load(f)


def _get_relevant_ids(q_data):
    """Extract relevant document IDs from qrels data."""
    if isinstance(q_data, dict):
        return set(str(k) for k in q_data.keys())
    elif isinstance(q_data, list):
        return set(str(k) for k in q_data)
    return set()


def _calculate_recall(relevant_ids: set, hit_ids: list, k: int) -> float:
    """Calculate Recall@k metric."""
    if not relevant_ids:
        return 0
    num_relevant_retrieved = len(relevant_ids.intersection(set(hit_ids)))
    return num_relevant_retrieved / min(k, len(relevant_ids))


def _calculate_mrr(relevant_ids: set, hit_ids: list) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""
    for i, hid in enumerate(hit_ids):
        if hid in relevant_ids:
            return 1 / (i + 1)
    return 0


def _dcg(rels):
    """Calculate Discounted Cumulative Gain for a list of relevance scores."""
    import math
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def _calculate_ndcg(qrels: dict, results: dict, k: int) -> list:
    """Calculate nDCG@k for all queries."""
    ndcgs = []
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        q_rel = qrels[qid]
        sorted_hits = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:k]
        rels = [q_rel.get(hid, 0) for hid, _ in sorted_hits]
        ideal_rels = sorted(q_rel.values(), reverse=True)[:k]
        ideal_d = _dcg(ideal_rels) if any(ideal_rels) else 0
        cur_d = _dcg(rels)
        ndcgs.append(cur_d / ideal_d if ideal_d > 0 else 0)
    return ndcgs


def _save_metrics(results_path: str, metrics: dict, k: int):
    """Save metrics to JSON file."""
    metrics_file = results_path.replace('.json', f'_metrics_k{k}.json')
    with open(metrics_file, 'w', encoding='utf-8') as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Metrics saved to {metrics_file}")


def calculate_metrics(results_path: str, qrels_path: str, k: int = 10):
    """Calculate BEIR evaluation metrics from retrieval results.

    Args:
        results_path: Path to JSON file containing retrieval results
        qrels_path: Path to qrels file (TSV or JSON format)
        k: Number of top results to evaluate (default: 10)
    """
    print(f"Loading results from {results_path}...")
    results = _load_results(results_path)

    print(f"Loading qrels from {qrels_path}...")
    qrels = _load_qrels(qrels_path)

    # Initialize metric accumulators
    recall_at_k = []
    mrr = []
    found_queries = 0

    for qid, hits in results.items():
        if qid not in qrels:
            continue

        found_queries += 1
        relevant_ids = _get_relevant_ids(qrels[qid])
        
        # Sort hits by score descending and take top-k
        sorted_hits = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:k]
        hit_ids = [str(hid) for hid, score in sorted_hits]

        # Calculate metrics
        recall_at_k.append(_calculate_recall(relevant_ids, hit_ids, k))
        mrr.append(_calculate_mrr(relevant_ids, hit_ids))

    if not found_queries:
        print("No matching queries found between results and qrels!")
        return

    avg_recall = sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0
    avg_mrr = sum(mrr) / len(mrr) if mrr else 0

    # Compute nDCG@k
    ndcgs = _calculate_ndcg(qrels, results, k)
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    # Display results
    print(f"\n--- Performance Evidence (k={k}) ---")
    print(f"Queries Evaluated: {found_queries}")
    print(f"Avg Recall@{k}: {avg_recall:.4f}")
    print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
    print(f"Avg nDCG@{k}: {avg_ndcg:.4f}")
    print("----------------------------------")

    # Save metrics to file
    metrics = {
        "queries_evaluated": found_queries,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
        "k": k,
    }
    _save_metrics(results_path, metrics, k)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate BEIR metrics from run results")
    parser.add_argument("--results", type=str, default="results/beir_run_nfcorpus.json", help="Path to the results JSON file")
    parser.add_argument("--qrels", type=str, default="data/beir/nfcorpus/qrels/test.tsv", help="Path to the qrels file (TSV or JSON)")
    parser.add_argument("--k", type=int, default=10, help="Top-k for metrics calculation")
    
    args = parser.parse_args()
    calculate_metrics(args.results, args.qrels, args.k)
