"""
DEPRECATED shim module. Use `src.cubo.retrieval.retriever.HybridRetriever` instead.
This file remains as a compatibility wrapper: it re-exports the canonical implementation
from `retriever.py` and emits a DeprecationWarning on import so callers can migrate.
"""

import warnings

warnings.warn(
    "src.cubo.retrieval.hybrid_retriever is deprecated; import HybridRetriever from cubo.retrieval.retriever instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cubo.retrieval.retriever import HybridRetriever  # re-export canonical implementation

__all__ = ["HybridRetriever"]
"""
Hybrid Retriever that combines sparse (BM25) and dense (FAISS) retrieval.
"""
from typing import Dict, List

from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.retrieval.fusion import rrf_fuse


class HybridRetriever:
    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        faiss_manager: FAISSIndexManager,
        embedding_generator: EmbeddingGenerator,
        documents: List[Dict],
    ):
        self.bm25_searcher = bm25_searcher
        self.faiss_manager = faiss_manager
        self.embedding_generator = embedding_generator
        self.documents = {doc["doc_id"]: doc for doc in documents}

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Performs hybrid search and returns a list of documents.
        """
        # 1. Get BM25 results
        bm25_results = self.bm25_searcher.search(query, top_k=top_k)

        # 2. Get FAISS results
        query_embedding = self.embedding_generator.encode([query])[0]
        faiss_results = self.faiss_manager.search(query_embedding, k=top_k)

        # 3. Fuse results (naive fusion)
        fused_results = self._fuse_results(bm25_results, faiss_results)

        # 4. Sort and return top-k
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        # Get the full document from the fused results
        final_results = []
        for res in fused_results[:top_k]:
            doc_id = res["doc_id"]
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                doc["score"] = res["score"]
                final_results.append(doc)

        return final_results

    def _fuse_results(self, bm25_results: List[Dict], faiss_results: List[Dict]) -> List[Dict]:
        # Use the shared rrf_fuse util for standardization across retrievers
        return rrf_fuse(bm25_results, faiss_results)
