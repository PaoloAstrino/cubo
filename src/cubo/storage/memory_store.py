"""
In-memory vector store fallback for testing and environments without binary dependencies.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.cubo.utils.trace_collector import trace_collector


class InMemoryCollection:
    """Lightweight in-memory collection fallback for tests and environments
    where binary dependencies cannot be imported."""

    def __init__(self):
        self._docs = []  # list of (id, document, metadata, embedding)

    def add(
        self,
        embeddings: Optional[List] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ):
        """Add documents to the collection."""
        if not documents:
            return

        for idx, doc in enumerate(documents):
            emb = None
            if embeddings:
                emb = embeddings[idx]
            meta = metadatas[idx] if metadatas else {}
            doc_id = ids[idx] if ids else f"doc_{len(self._docs)}"
            self._docs.append(
                (doc_id, doc, meta, np.asarray(emb, dtype="float32") if emb is not None else None)
            )
        if trace_id:
            try:
                trace_collector.record(
                    trace_id,
                    "vector_store",
                    "vector.added",
                    {"ids": ids or [], "count_added": len(documents) if documents else 0},
                )
            except Exception:
                pass

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return len(self._docs)

    def get(
        self,
        include: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get documents with optional filtering."""
        # Filter by ids or simple where clauses
        results = self._docs
        if ids:
            id_set = set(ids)
            results = [d for d in results if d[0] in id_set]
        if where and isinstance(where, dict):
            for key, val in where.items():
                if isinstance(val, dict) and "$in" in val:
                    # Handle $in operator
                    valid_values = set(val["$in"])
                    results = [d for d in results if d[2].get(key) in valid_values]
                else:
                    results = [d for d in results if d[2].get(key) == val]

        docs = [d[1] for d in results]
        metas = [d[2] for d in results]
        ids_out = [d[0] for d in results]
        return {"ids": ids_out, "documents": docs, "metadatas": metas}

    def query(
        self,
        query_embeddings: Optional[List] = None,
        n_results: int = 10,
        include: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query the collection using cosine similarity."""
        if not query_embeddings or not self._docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        q = np.asarray(query_embeddings[0], dtype="float32")

        # Apply where filter first
        filtered_docs = self._docs
        if where and isinstance(where, dict):
            for key, val in where.items():
                if isinstance(val, dict) and "$in" in val:
                    valid_values = set(val["$in"])
                    filtered_docs = [d for d in filtered_docs if d[2].get(key) in valid_values]
                else:
                    filtered_docs = [d for d in filtered_docs if d[2].get(key) == val]

        # Calculate similarities
        similarities = []
        for i, (doc_id, doc, meta, emb) in enumerate(filtered_docs):
            if emb is None:
                sim = 0.0
            else:
                denom = np.linalg.norm(emb) * np.linalg.norm(q)
                sim = float(np.dot(emb, q) / denom) if denom > 0 else 0.0
            similarities.append((i, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:n_results]

        docs = [filtered_docs[idx][1] for idx, _ in top]
        metas = [filtered_docs[idx][2] for idx, _ in top]
        dists = [1.0 - sim for _, sim in top]
        doc_ids = [filtered_docs[idx][0] for idx, _ in top]

        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [doc_ids]}

    def reset(self):
        """Clear all documents from the collection."""
        self._docs.clear()

    def close(self):
        """Close the in-memory collection. Provided for API parity with FaissStore.

        This is a no-op for the in-memory implementation but allows callers to
        use a uniform close() API across different backends.
        """
        self.reset()
