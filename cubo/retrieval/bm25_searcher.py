"""
Simple BM25 searcher for fast-pass JSONL output and stats.

This module provides BM25 ranking functionality with support for
multiple backend implementations (Python, Whoosh, etc.).
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from cubo.retrieval.bm25_store_factory import get_bm25_store
from cubo.retrieval.constants import BM25_B, BM25_K1, BM25_NORMALIZATION_FACTOR
from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer
from cubo.utils.logger import logger


class BM25Searcher:
    """Backward-compatible wrapper that delegates to a BM25Store implementation.

    It preserves the public API while allowing runtime backend selection via config
    or via the `backend` parameter.

    Attributes:
        chunks_jsonl: Path to chunks JSONL file.
        bm25_stats: Path to BM25 statistics JSON file.
        docs: List of indexed documents.
    """

    # BM25 tuning parameters (from centralized constants)
    K1 = BM25_K1
    B = BM25_B

    @property
    def docs(self) -> List[Dict]:
        """Delegate docs to the underlying store."""
        return getattr(self._store, "docs", [])

    @docs.setter
    def docs(self, value: List[Dict]):
        """Setter to maintain compatibility if something tries to write to it."""
        if hasattr(self._store, "docs"):
            self._store.docs = value

    def __init__(
        self, chunks_jsonl: str = None, bm25_stats: str = None, backend: str = None, **kwargs
    ):
        """Initialize BM25 searcher with optional data paths.

        Args:
            chunks_jsonl: Path to JSONL file with document chunks.
            bm25_stats: Path to JSON file with BM25 statistics.
            backend: Backend implementation name ('python', 'whoosh', etc.).
            **kwargs: Additional arguments passed to backend store.
        """
        self.chunks_jsonl = Path(chunks_jsonl) if chunks_jsonl else None
        self.bm25_stats = Path(bm25_stats) if bm25_stats else None
        self._store = get_bm25_store(backend=backend, **kwargs)

        # Initialize multilingual tokenizer
        try:
            self.tokenizer = MultilingualTokenizer()
        except Exception:
            self.tokenizer = None

        # Legacy fallback attributes to preserve wrapper behavior
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.term_doc_freq: Dict[str, int] = defaultdict(int)
        self.doc_term_freq: Dict[str, Dict[str, int]] = {}

        # Load data if paths provided
        if self.chunks_jsonl or self.bm25_stats:
            self._load()

    def _load(self) -> None:
        """Load chunks and BM25 statistics from configured paths."""
        docs_parsed = self._parse_chunks_file()
        self._index_parsed_docs(docs_parsed)
        self._load_bm25_stats()
        self.docs = getattr(self._store, "docs", [])

    def _parse_chunks_file(self) -> List[Dict[str, Any]]:
        """Parse chunks from JSONL file.

        Returns:
            List of parsed document dicts with doc_id, text, and metadata.
        """
        if not self.chunks_jsonl or not self.chunks_jsonl.exists():
            return []

        docs_parsed = []
        with open(self.chunks_jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = self._build_doc_id(rec)
                text = rec.get("text", rec.get("document", ""))
                docs_parsed.append({"doc_id": doc_id, "text": text, "metadata": rec})
        return docs_parsed

    def _build_doc_id(self, rec: Dict[str, Any]) -> str:
        """Build document ID from record metadata.

        Args:
            rec: Record dict with file_hash, chunk_index, or filename.

        Returns:
            Unique document ID string.
        """
        file_hash = rec.get("file_hash", "")
        chunk_index = rec.get("chunk_index", 0)
        filename = rec.get("filename", "unknown")

        if file_hash:
            return f"{file_hash}_{chunk_index}"
        return f"{filename}_{chunk_index}"

    def _index_parsed_docs(self, docs_parsed: List[Dict[str, Any]]) -> None:
        """Index parsed documents into the backend store.

        Args:
            docs_parsed: List of document dicts to index.
        """
        if not docs_parsed:
            return

        try:
            self._store.index_documents(docs_parsed)
        except Exception:
            self._try_add_documents_fallback(docs_parsed)

    def _try_add_documents_fallback(self, docs_parsed: List[Dict[str, Any]]) -> None:
        """Fallback: try incremental document addition.

        Args:
            docs_parsed: List of document dicts to add.
        """
        try:
            self._store.add_documents(docs_parsed)
        except Exception:
            logger.warning("Failed to index docs into BM25 store")

    def _load_bm25_stats(self) -> None:
        """Load BM25 statistics from configured stats file."""
        if not self.bm25_stats or not self.bm25_stats.exists():
            return

        try:
            self._store.load_stats(str(self.bm25_stats))
        except Exception:
            logger.warning("BM25 store failed to load stats")

    def load_stats(self, path: str) -> None:
        """Load BM25 statistics from a JSON file.

        Args:
            path: Path to the stats JSON file.
        """
        try:
            return self._store.load_stats(path)
        except Exception:
            self._load_stats_legacy(path)

    def _load_stats_legacy(self, path: str) -> None:
        """Legacy fallback for loading stats into store attributes.

        Args:
            path: Path to the stats JSON file.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._apply_stats_to_store(data)

    def _apply_stats_to_store(self, data: Dict[str, Any]) -> None:
        """Apply loaded stats data to the store attributes.

        Args:
            data: Dict containing doc_lengths, avg_doc_length, etc.
        """
        if hasattr(self._store, "doc_lengths"):
            self._store.doc_lengths = {k: int(v) for k, v in data.get("doc_lengths", {}).items()}
        if hasattr(self._store, "avg_doc_length"):
            self._store.avg_doc_length = float(data.get("avg_doc_length", 0.0))
        if hasattr(self._store, "term_doc_freq"):
            self._store.term_doc_freq = defaultdict(
                int, {k: int(v) for k, v in data.get("term_doc_freq", {}).items()}
            )
        if hasattr(self._store, "doc_term_freq"):
            self._store.doc_term_freq = {
                k: {tk: int(tv) for tk, tv in v.items()}
                for k, v in data.get("doc_term_freq", {}).items()
            }

    def save_stats(self, path: str) -> None:
        """Save BM25 stats to a JSON file.

        Args:
            path: Path for the output JSON file.
        """
        try:
            self._store.save_stats(path)
        except Exception:
            self._save_stats_legacy(path)

    def _save_stats_legacy(self, path: str) -> None:
        """Legacy fallback for saving stats from store attributes.

        Args:
            path: Path for the output JSON file.
        """
        data = {
            "doc_lengths": getattr(self._store, "doc_lengths", {}),
            "avg_doc_length": getattr(self._store, "avg_doc_length", 0.0),
            "term_doc_freq": dict(getattr(self._store, "term_doc_freq", {})),
            "doc_term_freq": {
                doc_id: dict(freqs)
                for doc_id, freqs in getattr(self._store, "doc_term_freq", {}).items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def index_documents(self, docs: List[Dict]) -> Any:
        """Index a list of documents.

        Args:
            docs: List of dicts with 'doc_id' and 'text' keys.

        Returns:
            Result from backend store's index_documents.
        """
        return self._store.index_documents(docs)

    def add_documents(self, docs: List[Dict], reset: bool = False) -> Any:
        """Add documents to the index incrementally.

        Args:
            docs: List of dicts with 'doc_id' and 'text' keys.
            reset: If True, clears existing stats before adding.

        Returns:
            Result from backend store or None.
        """
        try:
            return self._store.add_documents(docs, reset=reset)
        except Exception:
            return self._add_documents_fallback(docs, reset)

    def _add_documents_fallback(self, docs: List[Dict], reset: bool) -> None:
        """Fallback implementation for adding documents.

        Args:
            docs: List of document dicts.
            reset: If True, reindex all documents.
        """
        if reset:
            self.index_documents(docs)
            return

        for d in docs:
            doc_id = d.get("doc_id")
            if not doc_id:
                continue
            if d not in self.docs:
                self.docs.append(d)

            text = d.get("text", "")
            tokens = self._tokenize(text)
            self._update_doc_stats(doc_id, tokens)

        self._recalculate_avg_doc_length()

    def _update_doc_stats(self, doc_id: str, tokens: List[str]) -> None:
        """Update statistics for a single document.

        Args:
            doc_id: Document identifier.
            tokens: List of tokens from the document.
        """
        tf = Counter(tokens)
        self.doc_term_freq[doc_id] = dict(tf)
        self.doc_lengths[doc_id] = len(tokens)
        for term in set(tokens):
            self.term_doc_freq[term] += 1

    def _recalculate_avg_doc_length(self) -> None:
        """Recalculate average document length from current stats."""
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the multilingual tokenizer.

        Args:
            text: Input text string.

        Returns:
            List of stemmed/processed tokens.
        """
        if self.tokenizer:
            return self.tokenizer.tokenize(text)

        # Regex-based fallback (better than simple split)
        import re

        return [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2]

    def compute_score(self, query_terms: List[str], doc_id: str, doc_text: str = None) -> float:
        """Compute BM25 score for a document.

        Args:
            query_terms: List of query tokens.
            doc_id: Document identifier.
            doc_text: Optional document text for on-the-fly scoring.

        Returns:
            BM25 score as float.
        """
        try:
            return self._store.compute_score(query_terms, doc_id, doc_text)
        except Exception:
            return self._compute_bm25_score(query_terms, doc_id, doc_text)

    def _compute_bm25_score(
        self, query_terms: List[str], doc_id: str, doc_text: str = None
    ) -> float:
        """Compute BM25 score using the classic formula.

        BM25 formula: sum over query terms of:
            IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))

        Args:
            query_terms: List of query tokens.
            doc_id: Document identifier.
            doc_text: Optional document text.

        Returns:
            BM25 score as float.
        """
        doc_len = self._get_doc_length(doc_id, doc_text)
        if doc_len == 0:
            return 0.0

        total_docs = len(self.doc_lengths) if self.doc_lengths else 1
        avg_len = self.avg_doc_length if self.avg_doc_length > 0 else doc_len
        doc_terms = self._get_doc_terms(doc_id, doc_text)

        return self._sum_term_scores(query_terms, doc_terms, doc_len, avg_len, total_docs)

    def _get_doc_length(self, doc_id: str, doc_text: Optional[str]) -> int:
        """Get document length, computing from text if not cached.

        Args:
            doc_id: Document identifier.
            doc_text: Optional document text.

        Returns:
            Document length in tokens.
        """
        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0 and doc_text:
            return len(self._tokenize(doc_text))
        return doc_len

    def _get_doc_terms(self, doc_id: str, doc_text: Optional[str]) -> Dict[str, int]:
        """Get term frequencies for a document.

        Args:
            doc_id: Document identifier.
            doc_text: Optional document text.

        Returns:
            Dict mapping terms to their frequencies.
        """
        doc_terms = self.doc_term_freq.get(doc_id, {})
        if not doc_terms:
            text = doc_text or self._find_doc_text(doc_id)
            tokens = self._tokenize(text)
            doc_terms = dict(Counter(tokens))
        return doc_terms

    def _find_doc_text(self, doc_id: str) -> str:
        """Find document text by ID in the docs list.

        Args:
            doc_id: Document identifier to search for.

        Returns:
            Document text or empty string if not found.
        """
        return next((d["text"] for d in self.docs if d.get("doc_id") == doc_id), "")

    def _sum_term_scores(
        self,
        query_terms: List[str],
        doc_terms: Dict[str, int],
        doc_len: int,
        avg_len: float,
        total_docs: int,
    ) -> float:
        """Sum BM25 scores for all query terms.

        Args:
            query_terms: List of query tokens.
            doc_terms: Dict of term frequencies in document.
            doc_len: Document length.
            avg_len: Average document length.
            total_docs: Total number of documents.

        Returns:
            Total BM25 score.
        """
        score = 0.0
        for term in query_terms:
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue
            score += self._compute_term_score(term, tf, doc_len, avg_len, total_docs)
        return score

    def _compute_term_score(
        self, term: str, tf: int, doc_len: int, avg_len: float, total_docs: int
    ) -> float:
        """Compute BM25 score contribution from a single term.

        Args:
            term: The query term.
            tf: Term frequency in document.
            doc_len: Document length.
            avg_len: Average document length.
            total_docs: Total number of documents.

        Returns:
            BM25 score for this term.
        """
        df = self.term_doc_freq.get(term, 0)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        numerator = tf * (self.K1 + 1)
        denominator = tf + self.K1 * (1 - self.B + self.B * (doc_len / avg_len))
        return idf * (numerator / denominator)

    def search(self, query: str, top_k: int = 10, docs: List[Dict] = None) -> List[Dict]:
        """Search for documents matching query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            docs: Optional list of docs to search (default: all indexed).

        Returns:
            List of result dicts with doc_id, similarity, metadata, text.
        """
        try:
            return self._store.search(query, top_k=top_k, docs=docs)
        except Exception:
            return self._search_fallback(query, top_k, docs)

    def _search_fallback(self, query: str, top_k: int, docs: Optional[List[Dict]]) -> List[Dict]:
        """Fallback search implementation using wrapper scoring.

        Args:
            query: Search query string.
            top_k: Maximum results.
            docs: Optional docs list to search.

        Returns:
            Sorted list of result dicts.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        docs_to_search = docs if docs is not None else self.docs
        results = self._score_documents(query_terms, docs_to_search)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _score_documents(self, query_terms: List[str], docs_to_search: List[Dict]) -> List[Dict]:
        """Score all documents against query terms.

        Args:
            query_terms: Tokenized query terms.
            docs_to_search: List of documents to score.

        Returns:
            List of result dicts with positive scores.
        """
        results = []
        for d in docs_to_search:
            doc_id = d.get("doc_id")
            if not doc_id:
                continue
            bm25_score = self._compute_bm25_score(query_terms, doc_id)
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
        return results
