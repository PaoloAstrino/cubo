"""
Scaffold Retriever for semantic compression layer.

Retrieves compressed scaffold representations and expands them to underlying chunks
for efficient semantic retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cubo.utils.logger import logger


class ScaffoldRetriever:
    """
    Retrieves using semantic scaffolds as a compression layer.
    
    Scaffolds group semantically similar chunks, allowing:
    - Faster search over compressed representations
    - Semantic clustering for better results
    - Expansion to underlying chunks when needed
    """

    def __init__(
        self,
        scaffold_index: Optional[Any] = None,
        scaffold_mapping: Optional[Dict[str, List[str]]] = None,
        embedding_generator: Optional[Any] = None
    ):
        """
        Initialize scaffold retriever.
        
        Args:
            scaffold_index: FAISS index for scaffold embeddings
            scaffold_mapping: Map from scaffold_id to list of chunk_ids
            embedding_generator: Generator for query embeddings
        """
        self.scaffold_index = scaffold_index
        self.scaffold_mapping = scaffold_mapping or {}
        self.embedding_generator = embedding_generator
        self.scaffold_ids = list(scaffold_mapping.keys()) if scaffold_mapping else []

    def retrieve_scaffolds(
        self,
        query: str,
        top_k: int = 5,
        expand_to_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve scaffolds relevant to query.
        
        Args:
            query: Search query
            top_k: Number of scaffolds to retrieve
            expand_to_chunks: Whether to expand scaffolds to chunk IDs
            
        Returns:
            List of scaffold results with scores
        """
        if self.scaffold_index is None:
            logger.warning("Scaffold index not initialized")
            return []
        
        if self.embedding_generator is None:
            logger.warning("Embedding generator not available")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])
        
        # Search scaffold index
        try:
            distances, indices = self.scaffold_index.search(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Scaffold search failed: {e}")
            return []
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.scaffold_ids):
                continue
            
            scaffold_id = self.scaffold_ids[idx]
            
            result = {
                'scaffold_id': scaffold_id,
                'score': float(1 / (1 + dist)),  # Convert distance to similarity
                'rank': i
            }
            
            # Expand to underlying chunks if requested
            if expand_to_chunks and scaffold_id in self.scaffold_mapping:
                result['chunk_ids'] = self.scaffold_mapping[scaffold_id]
                result['num_chunks'] = len(result['chunk_ids'])
            
            results.append(result)
        
        return results

    def expand_scaffolds_to_chunks(
        self,
        scaffold_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Expand scaffold results to underlying chunk IDs.
        
        Args:
            scaffold_results: Results from retrieve_scaffolds()
            
        Returns:
            Flattened list of chunk IDs
        """
        chunk_ids = []
        for result in scaffold_results:
            scaffold_id = result['scaffold_id']
            if scaffold_id in self.scaffold_mapping:
                chunk_ids.extend(self.scaffold_mapping[scaffold_id])
        
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        
        return deduped

    def load_scaffold_index(
        self,
        scaffold_dir: Path,
        use_faiss: bool = True
    ) -> bool:
        """
        Load scaffold index and mappings from disk.
        
        Args:
            scaffold_dir: Directory containing scaffold data
            use_faiss: Whether to use FAISS index
            
        Returns:
            True if loaded successfully
        """
        scaffold_dir = Path(scaffold_dir)
        
        # Load scaffold mappings
        mapping_path = scaffold_dir / 'scaffold_mappings.json'
        if mapping_path.exists():
            import json
            with open(mapping_path, 'r') as f:
                self.scaffold_mapping = json.load(f)
            self.scaffold_ids = list(self.scaffold_mapping.keys())
            logger.info(f"Loaded {len(self.scaffold_mapping)} scaffold mappings")
        else:
            logger.warning(f"Scaffold mappings not found at {mapping_path}")
            return False
        
        # Load FAISS index if available
        if use_faiss:
            index_path = scaffold_dir / 'scaffold_index.faiss'
            if index_path.exists():
                try:
                    import faiss
                    self.scaffold_index = faiss.read_index(str(index_path))
                    logger.info(f"Loaded scaffold FAISS index with {self.scaffold_index.ntotal} vectors")
                except ImportError:
                    logger.warning("FAISS not available, using fallback")
                    return self._load_numpy_index(scaffold_dir)
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}")
                    return False
            else:
                logger.warning(f"Scaffold FAISS index not found at {index_path}")
                return self._load_numpy_index(scaffold_dir)
        else:
            return self._load_numpy_index(scaffold_dir)
        
        return True

    def _load_numpy_index(self, scaffold_dir: Path) -> bool:
        """
        Load scaffold embeddings as numpy array (fallback).
        
        Args:
            scaffold_dir: Directory containing scaffold data
            
        Returns:
            True if loaded successfully
        """
        embeddings_path = scaffold_dir / 'scaffold_embeddings.npy'
        if not embeddings_path.exists():
            logger.warning(f"Scaffold embeddings not found at {embeddings_path}")
            return False
        
        try:
            scaffold_embeddings = np.load(embeddings_path)
            # Create simple in-memory index
            self.scaffold_index = SimpleIndex(scaffold_embeddings)
            logger.info(f"Loaded scaffold embeddings: {scaffold_embeddings.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load scaffold embeddings: {e}")
            return False

    @property
    def is_ready(self) -> bool:
        """Check if scaffold retriever is ready for use."""
        return (
            self.scaffold_index is not None
            and self.scaffold_mapping is not None
            and len(self.scaffold_mapping) > 0
        )


class SimpleIndex:
    """Simple in-memory index for scaffolds (fallback when FAISS unavailable)."""
    
    def __init__(self, embeddings: np.ndarray):
        """Initialize with embeddings array."""
        self.embeddings = embeddings
        self.ntotal = len(embeddings)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query embedding(s)
            k: Number of results
            
        Returns:
            Tuple of (distances, indices)
        """
        # Compute cosine similarity
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        # Normalize
        query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(query_norm, emb_norm.T)
        
        # Convert to distances (lower is better)
        distances = 1 - similarities
        
        # Get top-k
        top_k = min(k, len(self.embeddings))
        indices = np.argsort(distances, axis=1)[:, :top_k]
        distances = np.take_along_axis(distances, indices, axis=1)
        
        return distances, indices


def create_scaffold_retriever_from_directory(
    scaffold_dir: str,
    embedding_generator: Optional[Any] = None
) -> ScaffoldRetriever:
    """
    Convenience function to create scaffold retriever from saved data.
    
    Args:
        scaffold_dir: Directory containing scaffold index and mappings
        embedding_generator: Optional embedding generator
        
    Returns:
        Initialized ScaffoldRetriever
    """
    retriever = ScaffoldRetriever(embedding_generator=embedding_generator)
    
    if not retriever.load_scaffold_index(Path(scaffold_dir)):
        logger.warning(f"Failed to load scaffold index from {scaffold_dir}")
    
    return retriever
