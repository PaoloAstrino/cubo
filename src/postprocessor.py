"""
Postprocessors for enhancing retrieval results in sentence window retrieval.
"""

from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class WindowReplacementPostProcessor:
    """Replaces single sentence text with full window context."""

    def __init__(self, target_metadata_key: str = "window"):
        self.target_metadata_key = target_metadata_key

    def postprocess_results(self, retrieval_results: List[Dict]) -> List[Dict]:
        """
        Replace document text with window context from metadata.
        """
        processed_results = []

        for result in retrieval_results:
            processed_result = result.copy()
            metadata = result.get('metadata', {})

            # Replace text with window if available
            if self.target_metadata_key in metadata:
                window_text = metadata[self.target_metadata_key]
                if window_text and len(window_text.strip()) > 0:
                    processed_result['document'] = window_text
                    logger.debug(f"Replaced sentence with window context ({len(window_text)} chars)")

            processed_results.append(processed_result)

        return processed_results

class LocalReranker:
    """Simple reranker using local embedding model."""

    def __init__(self, model, top_n: int = 2):
        self.model = model
        self.top_n = top_n

    def rerank(self, query: str, candidates: List[Dict], max_results: int = None) -> List[Dict]:
        """
        Rerank candidates using semantic similarity to query.
        
        Args:
            query: Search query
            candidates: List of candidate documents
            max_results: Maximum number of results to return (None = return all)
        """
        if not max_results:
            max_results = len(candidates)
            
        if len(candidates) <= max_results:
            return candidates

        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]

            # Calculate similarities and rerank
            scored_candidates = []
            for candidate in candidates:
                doc_text = candidate.get('document', '')
                if doc_text:
                    # Generate document embedding
                    doc_embedding = self.model.encode([doc_text])[0]

                    # Cosine similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    scored_candidates.append((candidate, similarity))

            # Sort by similarity (descending) and return top results
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            reranked = [candidate for candidate, _ in scored_candidates[:max_results]]

            logger.debug(f"Reranked {len(candidates)} candidates to top {max_results}")
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed, returning original candidates: {e}")
            return candidates[:max_results]

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)