"""Reusable mock classes for CUBO testing.

This module provides mock implementations of external services
to enable reliable, fast unit testing without external dependencies.
"""

from typing import Any, Dict, List

import numpy as np


class MockEmbeddingModel:
    """Mock embedding model for testing.

    Returns deterministic embeddings based on input text hash.
    """

    def __init__(self, embedding_dim: int = 384):
        """Initialize mock embedding model.

        Args:
            embedding_dim: Dimension of embedding vectors.
        """
        self.embedding_dim = embedding_dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Generate deterministic embeddings.

        Args:
            texts: List of text strings to embed.
            **kwargs: Additional arguments (ignored).

        Returns:
            Array of embedding vectors.
        """
        embeddings = []
        for text in texts:
            # Use hash for deterministic but different embeddings
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        return np.array(embeddings)


class MockVectorStore:
    """Mock in-memory vector store for testing."""

    def __init__(self):
        """Initialize mock vector store."""
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]], ids: List[str]):
        """Add vectors  to the store.

        Args:
            vectors: Embedding vectors to add.
            metadata: Metadata for each vector.
            ids: Unique IDs for each vector.
        """
        for vec, meta, doc_id in zip(vectors, metadata, ids):
            self.vectors.append(vec)
            self.metadata.append(meta)
            self.ids.append(doc_id)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results with metadata and scores.
        """
        if not self.vectors:
            return []

        # Compute cosine similarity
        scores = []
        for vec in self.vectors:
            similarity = np.dot(query_vector, vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vec)
            )
            scores.append(similarity)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {"id": self.ids[idx], "score": float(scores[idx]), "metadata": self.metadata[idx]}
            )

        return results


class MockLLMClient:
    """Mock LLM client for deterministic responses."""

    def __init__(self, default_response: str = "Mock LLM response"):
        """Initialize mock LLM client.

        Args:
            default_response: Default response text.
        """
        self.default_response = default_response
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a deterministic response.

        Args:
            prompt: Input prompt.
            **kwargs: Additional arguments (ignored).

        Returns:
            Mock response text.
        """
        self.call_count += 1
        return f"{self.default_response} (call #{self.call_count})"

    def generate_response(self, query: str, context: str) -> str:
        """Generate response given query and context.

        Args:
            query: User query.
            context: Context string.

        Returns:
            Mock response text.
        """
        return f"Answer to '{query}' based on context"


class MockWhooshIndex:
    """Mock Whoosh index wrapper for testing."""

    def __init__(self, index_dir: str):
        """Initialize mock Whoosh index.

        Args:
            index_dir: Index directory path (not actually used).
        """
        self.index_dir = index_dir
        self.documents: List[Dict[str, Any]] = []

    def index_documents(self, docs: List[Dict[str, Any]]):
        """Add documents to the mock index.

        Args:
            docs: List of documents to index.
        """
        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the mock index.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of matching documents.
        """
        # Simple mock: return documents containing query terms
        query_terms = query.lower().split()
        results = []

        for doc in self.documents:
            text = doc.get("text", "").lower()
            score = sum(term in text for term in query_terms)
            if score > 0:
                results.append({**doc, "score": score / len(query_terms)})

        # Sort by score and return top-k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
