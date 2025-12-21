
import json
import math
import argparse
from typing import Dict, List, Set

def calculate_mrr(qrels: Dict[str, Set[str]], results: Dict[str, Dict[str, float]]) -> float:
    mrr = 0.0
    for qid, relevant_docs in qrels.items():
        if qid not in results:
            continue
        sorted_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        for i, (doc_id, score) in enumerate(sorted_results):
            if doc_id in relevant_docs:
                mrr += 1.0 / (i + 1)
                break
    return mrr / len(qrels) if qrels else 0.0

def calculate_recall_at_k(qrels: Dict[str, Set[str]], results: Dict[str, Dict[str, float]], k: int) -> float:
    recall = 0.0
    for qid, relevant_docs in qrels.items():
        if qid not in results:
            continue
        sorted_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        found = 0
        for doc_id, score in sorted_results:
            if doc_id in relevant_docs:
                found += 1
        recall += found / len(relevant_docs) if relevant_docs else 0.0
    return recall / len(qrels) if qrels else 0.0

def dcg_at_k(relevances: List[int], k: int) -> float:
    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg

def calculate_ndcg_at_k(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]], k: int) -> float:
    ndcg = 0.0
    for qid, relevant_docs_scores in qrels.items():
        if qid not in results:
            continue
        
        # Predicted scores
        sorted_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        actual_relevances = [relevant_docs_scores.get(doc_id, 0) for doc_id, _ in sorted_results]
        
        # Ideal scores
        ideal_relevances = sorted(list(relevant_docs_scores.values()), reverse=True)[:k]
        
        actual_dcg = dcg_at_k(actual_relevances, k)
        ideal_dcg = dcg_at_k(ideal_relevances, k)
        
        if ideal_dcg > 0:
            ndcg += actual_dcg / ideal_dcg
            
    return ndcg / len(qrels) if qrels else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--qrels", type=str, required=True)
    args = parser.parse_args()

    with open(args.results, 'r') as f:
        results = json.load(f)

    qrels = {}
    with open(args.qrels, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            qid, did, score = line.strip().split('\t')
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(score)

    qrels_sets = {qid: set(docs.keys()) for qid, docs in qrels.items()}

    print(f"--- BEIR Benchmark Summary ({len(results)} queries) ---")
    
    ndcg10 = calculate_ndcg_at_k(qrels, results, 10)
    recall10 = calculate_recall_at_k(qrels_sets, results, 10)
    recall100 = calculate_recall_at_k(qrels_sets, results, 100)
    mrr = calculate_mrr(qrels_sets, results)

    print(f"NDCG@10:    {ndcg10:.4f}")
    print(f"Recall@10:  {recall10:.4f}")
    print(f"Recall@100: {recall100:.4f}")
    print(f"MRR:        {mrr:.4f}")

if __name__ == "__main__":
    main()
