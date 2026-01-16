import numpy as np
from typing import Dict, List, Tuple

def rrf_score(ranks: List[int], k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return sum(1.0 / (k + r) for r in ranks)

def weighted_sum_score(scores: List[float], weights: List[float]) -> float:
    """Normalized weighted sum score."""
    return sum(s * w for s, w in zip(scores, weights))

def borda_count_score(ranks: List[int], total_docs: int) -> float:
    """Borda Count score."""
    return sum(total_docs - r for r in ranks)

def compare_fusion_methods(dense_results: Dict[str, float], sparse_results: Dict[str, float]):
    """
    Compare RRF vs Weighted Sum vs Borda Count.
    Demonstrates why RRF is preferred for zero-tuning environments.
    """
    all_docs = list(set(dense_results.keys()) | set(sparse_results.keys()))
    
    # Sort for ranks
    dense_ranks = {doc: r for r, (doc, _) in enumerate(sorted(dense_results.items(), key=lambda x: x[1], reverse=True))}
    sparse_ranks = {doc: r for r, (doc, _) in enumerate(sorted(sparse_results.items(), key=lambda x: x[1], reverse=True))}
    
    methods = ["RRF (k=60)", "Weighted Sum (0.6/0.4)", "Borda Count"]
    print(f"{'Doc ID':<10} | {'Method':<20} | {'Score':<10} | {'Rank'}")
    print("-" * 60)
    
    for doc in all_docs:
        dr = dense_ranks.get(doc, 1000)
        sr = sparse_ranks.get(doc, 1000)
        ds = dense_results.get(doc, 0.0)
        ss = sparse_results.get(doc, 0.0)
        
        scores = {
            "RRF (k=60)": rrf_score([dr, sr]),
            "Weighted Sum (0.6/0.4)": weighted_sum_score([ds, ss], [0.6, 0.4]),
            "Borda Count": borda_count_score([dr, sr], 1000)
        }
        
        for name, score in scores.items():
             print(f"{doc:<10} | {name:<20} | {score:<10.4f}")
        print("-" * 60)

if __name__ == "__main__":
    # Example results with disparate distributions
    # Sparse (BM25) is unbounded, Dense (Cosine) is [0, 1]
    dense_ex = {"doc1": 0.92, "doc2": 0.88, "doc3": 0.45}
    sparse_ex = {"doc3": 24.5, "doc2": 15.2, "doc1": 2.1}
    
    print("DEMONSTRATION: Fusion Methods on Disparate Distributions")
    compare_fusion_methods(dense_ex, sparse_ex)
    print("\nCONCLUSION: RRF handles the 'scale mismatch' (0.92 vs 24.5) without needing normalization parameters.")
