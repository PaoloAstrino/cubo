"""
Retrieval Executor - Handles the actual retrieval operations.

This module encapsulates the core retrieval logic including:
- Dense (semantic) retrieval via vector store
- Sparse (BM25) retrieval
- Query embedding generation
- Result processing and scoring
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Set

from cubo.config import config

# Import the embedding prompt helper
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.retrieval.constants import (
    BM25_NORMALIZATION_FACTOR,
    BM25_WEIGHT_DETAILED,
    SEMANTIC_WEIGHT_DETAILED,
)
from cubo.utils.exceptions import DatabaseError
from cubo.utils.logger import logger


def extract_chunk_id(candidate: Any) -> Optional[str]:
    """Utility to extract a stable chunk/document id from a retrieval candidate.

    Many parts of the codebase expect a consistent callable for deduplication.
    This helper mirrors the common access pattern in result dicts.
    """
    if not isinstance(candidate, dict):
        return None
    # Top-level fields
    for k in ("id", "doc_id", "chunk_id"):
        val = candidate.get(k)
        if val:
            return str(val)
    # Fallback to metadata id
    meta = candidate.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("id", "doc_id", "chunk_id"):
            val = meta.get(k)
            if val:
                return str(val)
    return None


class RetrievalExecutor:
    """
    Executes retrieval operations against vector stores and BM25 indexes.

    This class handles the low-level mechanics of retrieval:
    - Generating query embeddings
    - Querying the vector store
    - Executing BM25 searches
    - Processing and scoring results
    """

    def __init__(
        self,
        collection: Any,
        bm25_searcher: Any,
        model: Any,
        inference_threading: Any,
        semantic_cache: Any = None,
    ):
        """
        Initialize the retrieval executor.

        Args:
            collection: Vector store collection
            bm25_searcher: BM25 searcher instance
            model: Embedding model
            inference_threading: Threading helper for embeddings
            semantic_cache: Optional semantic cache
        """
        self.collection = collection
        self.bm25 = bm25_searcher
        self.model = model
        self.inference_threading = inference_threading
        self.semantic_cache = semantic_cache

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Query string

        Returns:
            Embedding vector as list of floats
        """
        if self.model is None or self.inference_threading is None:
            return []

        # Apply query prompt prefix if model defines one
        try:
            prefix = EmbeddingGenerator.get_prompt_prefix_for_model(
                config.get("model_path"), "query"
            )
            prefixed_query = (prefix + query) if prefix else query
        except Exception:
            prefixed_query = query

        query_embeddings = self.inference_threading.generate_embeddings_threaded(
            [prefixed_query], self.model
        )
        return query_embeddings[0] if query_embeddings else []

    def query_dense(
        self,
        query_embedding: List[float],
        top_k: int,
        query: str = "",
        current_documents: Optional[Set[str]] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Execute dense (semantic) retrieval.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to retrieve
            query: Original query text for keyword boosting
            current_documents: Optional filter for document filenames
            trace_id: Optional trace ID for debugging

        Returns:
            List of candidate documents with scores
        """
        try:
            # Check cache first
            if self.semantic_cache:
                cached = self.semantic_cache.lookup(query_embedding, n_results=top_k)
                if cached:
                    logger.info("Semantic cache hit")
                    return cached

            # Build query params
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances", "ids"],
            }

            if current_documents:
                query_params["where"] = {"filename": {"$in": list(current_documents)}}

            if trace_id is not None:
                query_params["trace_id"] = trace_id

            results = self.collection.query(**query_params)
            try:
                raw_count = len(results.get("documents", [[]])[0]) if results else 0
                logger.debug(
                    f"dense retrieval requested={top_k}, raw_returned={raw_count}, trace_id={trace_id}"
                )
            except Exception:
                pass
            # Faiss / backend may return fewer results than requested. If so, pad
            # with additional documents from the collection (with base similarity
            # 0.0) up to 'top_k' to ensure deterministic result lengths.
            try:
                returned_docs = results.get("documents", [[]])[0]
                returned_ids = results.get("ids", [[]])[0]
                if len(returned_docs) < top_k:
                    logger.debug(
                        f"dense retrieval padding: have={len(returned_docs)}, need={top_k}"
                    )
                    all_docs = self.collection.get(include=["documents", "metadatas", "ids"]) or {}
                    all_ids = all_docs.get("ids", [])
                    all_metas = all_docs.get("metadatas", [])
                    all_docs_list = list(zip(all_ids, all_metas))
                    # Append until we reach top_k results
                    for doc_id, meta in all_docs_list:
                        if doc_id in returned_ids:
                            continue
                        returned_docs.append("")
                        # distances fill with 1.0 (similarity 0.0)
                        results.setdefault("distances", [[]])[0].append(1.0)
                        results.setdefault("ids", [[]])[0].append(doc_id)
                        results.setdefault("metadatas", [[]])[0].append(meta)
                        if len(returned_docs) >= top_k:
                            break
            except Exception:
                pass
            processed = self._process_dense_results(results, query)

            try:
                logger.debug(f"dense retrieval final_count={len(processed)}, trace_id={trace_id}")
            except Exception:
                pass

            if self.semantic_cache and processed:
                self.semantic_cache.add(query, query_embedding, processed)

            return processed

        except Exception as e:
            error_msg = f"Failed to query collection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg, "QUERY_FAILED", {}) from e

    def _process_dense_results(self, results: Dict, query: str = "") -> List[Dict]:
        """Process raw query results into candidate format with scoring."""
        candidates = []

        if (
            not results.get("documents")
            or not results.get("metadatas")
            or not results.get("distances")
        ):
            return candidates

        ids = results.get("ids", [[]])[0] if "ids" in results else []

        for i, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            # Ensure doc text is string; some vector stores return list-of-strings
            doc_text = self._to_text(doc)
            base_similarity = 1 - distance
            score_breakdown = self.compute_hybrid_score(doc, query, base_similarity)

            updated_metadata = (
                metadata.copy() if isinstance(metadata, dict) else {"metadata": metadata}
            )
            updated_metadata["score_breakdown"] = score_breakdown

            # Add document ID to metadata for IR metrics
            doc_id = ids[i] if i < len(ids) else None
            if doc_id:
                updated_metadata["id"] = doc_id
                updated_metadata["doc_id"] = doc_id

            candidates.append(
                {
                    "id": doc_id,
                    "document": doc_text,
                    "metadata": updated_metadata,
                    "similarity": score_breakdown["final_score"],
                    "base_similarity": base_similarity,
                }
            )

        return candidates

    def query_bm25(
        self,
        query: str,
        top_k: int,
        current_documents: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        Execute BM25 (sparse) retrieval.

        Args:
            query: Query string
            top_k: Number of results to retrieve
            current_documents: Optional filter for document filenames

        Returns:
            List of candidate documents with BM25 scores
        """
        try:
            # Allow disabling BM25 for faster dense-only retrieval in benchmarks
            bm25_enabled = config.get("retrieval.bm25_enabled", True)
            if isinstance(bm25_enabled, str):
                bm25_enabled = bm25_enabled.lower() not in ("0", "false", "no", "off")
            if not bm25_enabled:
                return []
            # If BM25 store already has an index precomputed (docs loaded), prefer
            # searching the internal store for performance.
            # Optimized path: if we have a filter, pass it as a set of IDs to the store.
            # This allows the store to use set intersection with posting lists (O(K))
            # instead of linearly scanning a filtered list of docs (O(N)).
            bm25_docs = getattr(self.bm25, "docs", [])

            if not current_documents and bm25_docs:
                # Fast path: full corpus search
                results = self.bm25.search(query, top_k=top_k)
            elif current_documents and bm25_docs:
                # Optimized path: Filtered search using inverted index intersection
                # We pass the set of allowed IDs directly.
                target_ids = set(current_documents)
                results = self.bm25.search(query, top_k=top_k, doc_ids=target_ids)
            else:
                # Fallback: retrieve explicit document list from the collection
                # This path is only taken if BM25 index is empty/not built.
                where_filter = (
                    {"filename": {"$in": list(current_documents)}} if current_documents else None
                )
                all_docs = self.collection.get(
                    include=["documents", "metadatas", "ids"],
                    where=where_filter,
                )
                if not all_docs.get("documents"):
                    return []
                # If BM25 store has not been populated yet, index all documents
                if not getattr(self.bm25, "docs", None):
                    docs_parsed = []
                    for i, (doc, metadata) in enumerate(
                        zip(all_docs["documents"], all_docs["metadatas"])
                    ):
                        doc_text = self._to_text(doc)
                        doc_id = (
                            all_docs.get("ids", [])[i]
                            if i < len(all_docs.get("ids", []))
                            else hashlib.md5(doc_text.encode(), usedforsecurity=False).hexdigest()[
                                :8
                            ]
                        )
                        docs_parsed.append(
                            {"doc_id": doc_id, "text": doc_text, "metadata": metadata}
                        )
                    try:
                        self.bm25.index_documents(docs_parsed)
                    except Exception:
                        # Fall back to adding incrementally on failure
                        try:
                            self.bm25.add_documents(docs_parsed)
                        except Exception:
                            pass
                # Build subset of docs for BM25 search if a filter exists
                docs_for_search = []
                ids_list = all_docs.get("ids", [])
                for i, (doc, metadata) in enumerate(
                    zip(all_docs["documents"], all_docs["metadatas"])
                ):
                    doc_text = self._to_text(doc)
                    doc_id = (
                        ids_list[i]
                        if i < len(ids_list)
                        else hashlib.md5(doc_text.encode(), usedforsecurity=False).hexdigest()[:8]
                    )
                    docs_for_search.append(
                        {
                            "doc_id": doc_id,
                            "text": doc_text,
                            "metadata": metadata,
                        }
                    )
                results = self.bm25.search(query, top_k=top_k, docs=docs_for_search)

            try:
                logger.debug(
                    f"bm25 retrieval docs={len(docs_for_search)}, returned={len(results)}, requested={top_k}"
                )
            except Exception:
                pass

            return [
                {
                    "id": r["doc_id"],
                    "document": r["text"],
                    "metadata": r["metadata"],
                    "similarity": r["similarity"],
                    "base_similarity": 0.0,
                    "bm25_score": r["similarity"] * BM25_NORMALIZATION_FACTOR,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []

    def compute_hybrid_score(
        self,
        document: str,
        query: str,
        base_similarity: float,
    ) -> Dict[str, float]:
        """
        Compute hybrid score combining semantic and BM25 scores.

        Args:
            document: Document text
            query: Query text
            base_similarity: Base semantic similarity score

        Returns:
            Dictionary with detailed score breakdown
        """
        if not query or not document:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0,
            }

        query_terms = self._tokenize(query)
        if not query_terms:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0,
            }

        # Ensure document is a string for hashing and BM25 scoring
        doc_text = self._to_text(document)
        doc_id = hashlib.md5(doc_text.encode(), usedforsecurity=False).hexdigest()[:8]
        bm25_score = self.bm25.compute_score(query_terms, doc_id, doc_text)
        normalized_bm25 = min(bm25_score / BM25_NORMALIZATION_FACTOR, 1.0)

        semantic_contribution = SEMANTIC_WEIGHT_DETAILED * base_similarity
        bm25_contribution = BM25_WEIGHT_DETAILED * normalized_bm25
        final_score = min(semantic_contribution + bm25_contribution, 1.0)

        return {
            "final_score": final_score,
            "semantic_score": base_similarity,
            "bm25_score": bm25_score,
            "semantic_contribution": semantic_contribution,
            "bm25_contribution": bm25_contribution,
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words using MultilingualTokenizer."""
        try:
            from cubo.retrieval.multilingual_tokenizer import tokenize_multilingual

            # Auto-detect language and stem
            return tokenize_multilingual(text, language="auto", use_stemming=True)
        except ImportError:
            # Fallback if dependencies missing
            words = re.findall(r"\b\w+\b", text.lower())
            return [w for w in words if len(w) > 2]

    def _to_text(self, doc: Any) -> str:
        """Converts a document payload into a plain text string.

        The vector store may return a single string or a list/tuple of strings.
        We coerce all non-str inputs into string joined by spaces which is
        sufficient for tokenization, hashing and BM25 scoring.
        """
        if doc is None:
            return ""
        if isinstance(doc, str):
            return doc
        if isinstance(doc, (list, tuple)):
            # Join list-like document parts into a single string
            try:
                return " ".join(map(str, doc))
            except Exception:
                return " ".join([str(x) for x in doc])
        # Fallback to stringifying other types
        return str(doc)
