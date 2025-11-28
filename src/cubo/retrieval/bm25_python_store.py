"""
Python BM25 store implementation for text retrieval.

This module provides a pure-Python BM25 (Best Matching 25) implementation
for document scoring and retrieval. BM25 is a ranking function used by
search engines to rank matching documents according to their relevance
to a given search query.

The implementation uses the standard BM25 formula with k1=1.5 and b=0.75
parameters, which work well for most retrieval tasks.
"""

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from src.cubo.retrieval.bm25_store import BM25Store
from src.cubo.retrieval.constants import BM25_B, BM25_K1, BM25_NORMALIZATION_FACTOR


class BM25PythonStore(BM25Store):
    """
    Pure-Python BM25 implementation for document retrieval.

    Maintains an inverted index of term frequencies and document lengths
    for efficient BM25 scoring. Supports incremental document addition
    and persistence to disk.

    Attributes:
        docs: List of indexed document dictionaries.
        doc_lengths: Mapping of doc_id to document length (token count).
        avg_doc_length: Average document length across corpus.
        term_doc_freq: Mapping of term to document frequency (df).
        doc_term_freq: Mapping of doc_id to term frequency dict.
    """

    # BM25 hyperparameters (from centralized constants)
    K1 = BM25_K1  # Term frequency saturation parameter
    B = BM25_B  # Document length normalization parameter

    def __init__(self, index_dir: Optional[str] = None, **kwargs):
        """Initialize empty BM25 index.

        Args:
            index_dir: Optional directory for persisting index stats.
                       Currently stored for reference but not auto-loaded.
            **kwargs: Additional arguments (ignored for forward compatibility).
        """
        self.index_dir = index_dir
        self.docs = []
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.term_doc_freq = defaultdict(int)  # term -> number of docs containing term
        self.doc_term_freq = {}  # doc_id -> {term: freq}

    def index_documents(self, docs: List[Dict]) -> None:
        """
        Build index from scratch with given documents.

        Clears existing index and rebuilds with new documents.

        Args:
            docs: List of document dicts with 'doc_id' and 'text' keys.
        """
        self.docs = docs
        self.doc_lengths = {}
        self.term_doc_freq = defaultdict(int)
        self.doc_term_freq = {}
        self.add_documents(docs, reset=False)

    def add_documents(self, docs: List[Dict], reset: bool = False) -> None:
        """
        Add documents to the index incrementally.

        Updates term frequencies and document statistics. Documents
        without 'doc_id' are skipped.

        Args:
            docs: List of document dicts with 'doc_id' and 'text' keys.
            reset: If True, clear index and rebuild from scratch.
        """
        if reset:
            self.index_documents(docs)
            return

        for d in docs:
            doc_id = d.get("doc_id")
            if not doc_id:
                continue

            # Add to docs list if not already present
            if d not in self.docs:
                self.docs.append(d)

            # Tokenize and compute term frequencies
            text = d.get("text", "")
            tokens = self._tokenize(text)
            tf = Counter(tokens)

            # Update index structures
            self.doc_term_freq[doc_id] = dict(tf)
            self.doc_lengths[doc_id] = len(tokens)

            # Update document frequencies for each unique term
            for term in set(tokens):
                self.term_doc_freq[term] += 1

        # Recompute average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase alphanumeric tokens.

        Filters out tokens shorter than 3 characters to remove
        common stopwords and noise.

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase tokens with length > 2.
        """
        # Replace non-alphanumeric with spaces, lowercase, split
        normalized = "".join(c if c.isalnum() else " " for c in text.lower())
        tokens = [w for w in normalized.split() if len(w) > 2]
        return tokens

    def compute_score(
        self, query_terms: List[str], doc_id: str, doc_text: Optional[str] = None
    ) -> float:
        """
        Compute BM25 score for a document given query terms.

        Args:
            query_terms: Tokenized query terms.
            doc_id: Document identifier.
            doc_text: Optional document text for on-the-fly scoring.

        Returns:
            BM25 relevance score (higher is more relevant).
        """
        return self._compute_bm25_score(query_terms, doc_id, doc_text)

    def _compute_bm25_score(
        self, query_terms: List[str], doc_id: str, doc_text: Optional[str] = None
    ) -> float:
        """
        Internal BM25 scoring implementation.

        Uses the BM25 formula:
        score = sum over terms of: IDF(t) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))

        Args:
            query_terms: Tokenized query terms.
            doc_id: Document identifier.
            doc_text: Optional text for documents not in index.

        Returns:
            BM25 score for the document.
        """
        # Get document length
        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            if doc_text:
                tokens = self._tokenize(doc_text)
                doc_len = len(tokens)
            else:
                return 0.0

        # Corpus statistics
        total_docs = len(self.doc_lengths) if self.doc_lengths else 1
        avg_len = self.avg_doc_length if self.avg_doc_length > 0 else doc_len

        # Get term frequencies for this document
        doc_terms = self.doc_term_freq.get(doc_id, {})
        if not doc_terms and doc_text is not None:
            tokens = self._tokenize(doc_text)
            doc_terms = dict(Counter(tokens))

        # Compute BM25 score
        score = 0.0
        for term in query_terms:
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue

            # Inverse document frequency with smoothing
            df = self.term_doc_freq.get(term, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score with saturation and length normalization
            numerator = tf * (self.K1 + 1)
            denominator = tf + self.K1 * (1 - self.B + self.B * (doc_len / avg_len))
            term_score = idf * (numerator / denominator)

            score += term_score

        return score

    def search(self, query: str, top_k: int = 10, docs: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Search for documents matching a query.

        Tokenizes query, scores all documents, and returns top-k results
        sorted by relevance.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            docs: Optional document subset to search within.

        Returns:
            List of result dicts with 'doc_id', 'similarity', 'metadata', 'text'.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        docs_to_search = docs if docs is not None else self.docs
        results = []

        for d in docs_to_search:
            doc_id = d.get("doc_id")
            if not doc_id:
                continue

            bm25_score = self._compute_bm25_score(query_terms, doc_id)

            # Normalize score to [0, 1] range using empirical max
            norm_score = min(bm25_score / BM25_NORMALIZATION_FACTOR, 1.0)

            if norm_score > 0.0:
                results.append(
                    {
                        "doc_id": doc_id,
                        "similarity": norm_score,
                        "metadata": d.get("metadata", {}),
                        "text": d.get("text", ""),
                    }
                )

        # Sort by score descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def load_stats(self, path: str) -> None:
        """
        Load index statistics from JSON file.

        Restores doc_lengths, avg_doc_length, term_doc_freq, and
        doc_term_freq from a previously saved state.

        Args:
            path: Path to JSON file with saved statistics.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            self.doc_lengths = {k: int(v) for k, v in data.get("doc_lengths", {}).items()}
            self.avg_doc_length = float(data.get("avg_doc_length", 0.0))
            self.term_doc_freq = defaultdict(
                int, {k: int(v) for k, v in data.get("term_doc_freq", {}).items()}
            )
            self.doc_term_freq = {
                k: {tk: int(tv) for tk, tv in v.items()}
                for k, v in data.get("doc_term_freq", {}).items()
            }

    def save_stats(self, path: str) -> None:
        """
        Save index statistics to JSON file.

        Persists doc_lengths, avg_doc_length, term_doc_freq, and
        doc_term_freq for later restoration.

        Args:
            path: Path to write JSON statistics file.
        """
        data = {
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "term_doc_freq": dict(self.term_doc_freq),
            "doc_term_freq": {doc_id: dict(freqs) for doc_id, freqs in self.doc_term_freq.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def close(self) -> None:
        """
        Clean up resources.

        No-op for in-memory Python store, but required by BM25Store interface.
        """
        return
