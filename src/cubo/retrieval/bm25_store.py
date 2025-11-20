"""
BM25 store plugin interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BM25Store(ABC):
    """Abstract base class for BM25 store implementations.

    Implementations should provide indexing and search APIs used by the application.
    """

    @abstractmethod
    def index_documents(self, docs: List[Dict]):
        """Build BM25 stats from a list of docs (replaces existing index)."""

    @abstractmethod
    def add_documents(self, docs: List[Dict], reset: bool = False):
        """Add documents to the index incrementally."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10, docs: Optional[List[Dict]] = None) -> List[Dict]:
        """Search for the query and return top_k results."""

    @abstractmethod
    def compute_score(self, query_terms: List[str], doc_id: str, doc_text: Optional[str] = None) -> float:
        """Compute BM25 score for a doc given query terms."""

    @abstractmethod
    def load_stats(self, path: str):
        """Load BM25 stats (for stat-based backends)."""

    @abstractmethod
    def save_stats(self, path: str):
        """Save BM25 stats (for stat-based backends)."""

    @abstractmethod
    def close(self):
        """Close or cleanup resources used by the store (if any)."""

    # Optional metadata: convenience attribute for docs
    docs: List[Dict] = []
