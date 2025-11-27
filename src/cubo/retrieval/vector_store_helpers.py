def _get_embedding_vector(self, doc_id: str) -> List[float]:
    """Get embedding vector for a document ID.

    Works in both modes:
    - Memory mode: Returns vector from _embeddings dict
    - Mmap mode: Looks up index and returns vector from mmap file

    Args:
        doc_id: Document ID

    Returns:
        Embedding vector as list of floats
    """
    if not self._mmap_mode:
        # Memory mode: direct lookup
        return self._embeddings.get(doc_id, [])

    # Mmap mode: lookup index, then get from mmap
    idx = self._embeddings.get(doc_id)
    if idx is None or self._vectors_mmap is None:
        return []

    # Return vector from memory-mapped array
    return self._vectors_mmap[idx].tolist()
