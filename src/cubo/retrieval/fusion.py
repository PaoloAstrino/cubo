"""
Fusion utilities for retrieval: RRF fusion and semantic + BM25 combiner.
"""
from typing import List, Dict


def rrf_fuse(bm25_results: List[Dict], faiss_results: List[Dict], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion: combine two ranked lists into fused scores.

    Both lists are expected to follow the shape: [{'doc_id': '...', 'score': float}] or
    FAISS can have {'id': '...','score': ...}. This function normalizes keys and returns
    a list of {doc_id, score}.
    """
    fused: Dict[str, Dict] = {}

    for i, doc in enumerate(bm25_results):
        rank = i + 1
        doc_id = doc.get('doc_id') or doc.get('id')
        if doc_id not in fused:
            fused[doc_id] = {'doc_id': doc_id, 'score': 0.0}
        fused[doc_id]['score'] += 1 / (k + rank)

    for i, doc in enumerate(faiss_results):
        rank = i + 1
        doc_id = doc.get('doc_id') or doc.get('id')
        if doc_id not in fused:
            fused[doc_id] = {'doc_id': doc_id, 'score': 0.0}
        fused[doc_id]['score'] += 1 / (k + rank)

    return list(fused.values())


def combine_semantic_and_bm25(semantic_candidates: List[Dict], bm25_candidates: List[Dict],
                             semantic_weight: float = 0.1, bm25_weight: float = 0.9,
                             top_k: int = 10) -> List[Dict]:
    """Combine semantic and BM25 candidate lists into a normalized combined ranking.

    Each candidate is expected as {'document': str, 'metadata': dict, 'similarity': float}
    """
    combined = {}
    for cand in semantic_candidates:
        doc_key = cand['document'][:100]
        if doc_key not in combined:
            combined[doc_key] = {
                'document': cand['document'],
                'metadata': cand.get('metadata', {}),
                'semantic_score': cand.get('similarity', 0.0),
                'bm25_score': 0.0,
            }
        else:
            combined[doc_key]['semantic_score'] = max(combined[doc_key]['semantic_score'], cand.get('similarity', 0.0))

    for cand in bm25_candidates:
        doc_key = cand['document'][:100]
        if doc_key not in combined:
            combined[doc_key] = {
                'document': cand['document'],
                'metadata': cand.get('metadata', {}),
                'semantic_score': 0.0,
                'bm25_score': cand.get('similarity', 0.0),
            }
        else:
            combined[doc_key]['bm25_score'] = max(combined[doc_key]['bm25_score'], cand.get('similarity', 0.0))

    final_results = []
    for doc_data in combined.values():
        combined_score = semantic_weight * doc_data['semantic_score'] + bm25_weight * doc_data['bm25_score']
        final_results.append({
            'document': doc_data['document'],
            'metadata': doc_data.get('metadata', {}),
            'similarity': combined_score,
            'base_similarity': doc_data['semantic_score'],
            'bm25_normalized': doc_data['bm25_score']
        })

    final_results.sort(key=lambda x: x['similarity'], reverse=True)
    return final_results[:top_k]
