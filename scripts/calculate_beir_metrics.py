import json
import os


def calculate_metrics(results_path: str, qrels_path: str, k: int = 10):
    print(f"Loading results from {results_path}...")
    with open(results_path, "r") as f:
        data = json.load(f)

    # Filter out metadata
    results = {k: v for k, v in data.items() if not k.startswith("_")}

    print(f"Loading qrels from {qrels_path}...")
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
    else:
        with open(qrels_path, "r") as f:
            qrels = json.load(f)

    recall_at_k = []
    mrr = []

    found_queries = 0
    for qid, hits in results.items():
        if qid not in qrels:
            continue

        found_queries += 1
        # Handle dict/list qrels format
        q_data = qrels[qid]
        relevant_ids = set()
        if isinstance(q_data, dict):
            relevant_ids = set(str(k) for k in q_data.keys())
        elif isinstance(q_data, list):
            relevant_ids = set(str(k) for k in q_data)

        # Sort hits by score descending
        sorted_hits = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:k]
        hit_ids = [str(hid) for hid, score in sorted_hits]

        # Recall@k
        num_relevant_retrieved = len(relevant_ids.intersection(set(hit_ids)))
        if len(relevant_ids) > 0:
            recall_at_k.append(num_relevant_retrieved / min(k, len(relevant_ids)))

        # MRR
        mrr_val = 0
        for i, hid in enumerate(hit_ids):
            if hid in relevant_ids:
                mrr_val = 1 / (i + 1)
                break
        mrr.append(mrr_val)

    if not found_queries:
        print("No matching queries found between results and qrels!")
        return

    avg_recall = sum(recall_at_k) / len(recall_at_k) if recall_at_k else 0
    avg_mrr = sum(mrr) / len(mrr) if mrr else 0

    # Compute nDCG@k
    def dcg(rels):
        import math
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    ndcgs = []
    for qid, hits in results.items():
        if qid not in qrels:
            continue
        # Build relevance list for top-k
        q_rel = qrels[qid]
        sorted_hits = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:k]
        rels = [q_rel.get(hid, 0) for hid, _ in sorted_hits]
        ideal_rels = sorted(q_rel.values(), reverse=True)[:k]
        ideal_d = dcg(ideal_rels) if any(ideal_rels) else 0
        cur_d = dcg(rels)
        ndcgs.append(cur_d / ideal_d if ideal_d > 0 else 0)

    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0

    print(f"\n--- Performance Evidence (k={k}) ---")
    print(f"Queries Evaluated: {found_queries}")
    print(f"Avg Recall@{k}: {avg_recall:.4f}")
    print(f"Mean Reciprocal Rank: {avg_mrr:.4f}")
    print(f"Avg nDCG@{k}: {avg_ndcg:.4f}")
    print(f"----------------------------------")

    # Save metrics to file
    metrics = {
        "queries_evaluated": found_queries,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
        "k": k,
    }

    metrics_file = results_path.replace('.json', f'_metrics_k{k}.json')
    with open(metrics_file, 'w', encoding='utf-8') as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate BEIR metrics from run results")
    parser.add_argument("--results", type=str, default="results/beir_run_nfcorpus.json", help="Path to the results JSON file")
    parser.add_argument("--qrels", type=str, default="data/beir/nfcorpus/qrels/test.tsv", help="Path to the qrels file (TSV or JSON)")
    parser.add_argument("--k", type=int, default=10, help="Top-k for metrics calculation")
    
    args = parser.parse_args()
    calculate_metrics(args.results, args.qrels, args.k)
