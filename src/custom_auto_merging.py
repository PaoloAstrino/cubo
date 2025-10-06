"""
Custom Auto-Merging Retrieval for CUBO
Implements hierarchical chunking and intelligent merging without LlamaIndex.
"""

import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import chromadb

from src.logger import logger
from src.config import config
from src.utils import Utils


class HierarchicalChunker:
    """Creates hierarchical chunks at multiple levels."""

    def __init__(self, chunk_sizes: List[int] = None):
        if chunk_sizes is None:
            # Use config values or defaults
            chunk_sizes = config.get("auto_merging_chunk_sizes", [2048, 512, 128])
        self.chunk_sizes = chunk_sizes

    def create_hierarchical_chunks(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Create hierarchical chunks from text."""
        chunks = []

        # Create chunks at each level
        for level, chunk_size in enumerate(self.chunk_sizes):
            level_chunks = self._create_chunks_at_level(text, chunk_size, level, filename)
            chunks.extend(level_chunks)

        return chunks

    def _create_chunks_at_level(self, text: str, chunk_size: int, level: int, filename: str) -> List[Dict[str, Any]]:
        """Create chunks at a specific hierarchical level."""
        chunks = []

        # Split text into chunks of approximately chunk_size tokens
        words = text.split()
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = len(word.split())  # Approximate token count

            if current_tokens + word_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(filename, level, len(chunks))

                chunk = {
                    'id': chunk_id,
                    'text': chunk_text,
                    'filename': filename,
                    'level': level,
                    'chunk_size': chunk_size,
                    'token_count': current_tokens,
                    'start_pos': len(' '.join(words[:len(current_chunk)])),
                    'end_pos': len(' '.join(words[:len(current_chunk) + len(current_chunk)])),
                    'parent_id': None,  # Will be set later
                    'child_ids': []
                }
                chunks.append(chunk)

                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(filename, level, len(chunks))

            chunk = {
                'id': chunk_id,
                'text': chunk_text,
                'filename': filename,
                'level': level,
                'chunk_size': chunk_size,
                'token_count': current_tokens,
                'start_pos': len(' '.join(words[:len(current_chunk)])),
                'end_pos': len(text),
                'parent_id': None,
                'child_ids': []
            }
            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, filename: str, level: int, index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{filename}_{level}_{index}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]

    def build_hierarchy(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build parent-child relationships between chunks."""
        # Group chunks by level
        level_groups = self._group_chunks_by_level(chunks)

        # Establish parent-child relationships for each level
        self._establish_parent_child_relationships(level_groups)

        return chunks

    def _group_chunks_by_level(self, chunks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group chunks by their hierarchical level.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Dictionary mapping level numbers to lists of chunks
        """
        level_groups = defaultdict(list)
        for chunk in chunks:
            level_groups[chunk['level']].append(chunk)
        return level_groups

    def _establish_parent_child_relationships(self, level_groups: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Establish parent-child relationships between chunks at different levels.

        Args:
            level_groups: Dictionary mapping level numbers to lists of chunks
        """
        for level in range(len(self.chunk_sizes) - 1):  # Don't process leaf level
            parent_level = level
            child_level = level + 1

            if parent_level in level_groups and child_level in level_groups:
                parents = level_groups[parent_level]
                children = level_groups[child_level]

                # Assign children to parents based on position overlap
                for parent in parents:
                    self._assign_children_to_parent(parent, children)

    def _assign_children_to_parent(self, parent: Dict[str, Any], children: List[Dict[str, Any]]) -> None:
        """
        Assign child chunks to a parent chunk based on position overlap.

        Args:
            parent: Parent chunk dictionary
            children: List of potential child chunks
        """
        parent_children = []
        parent_start = parent['start_pos']
        parent_end = parent['end_pos']

        for child in children:
            child_start = child['start_pos']
            child_end = child['end_pos']

            # Check if child overlaps with parent
            if self._chunks_overlap(parent_start, parent_end, child_start, child_end):
                parent_children.append(child['id'])
                child['parent_id'] = parent['id']

        parent['child_ids'] = parent_children

    def _chunks_overlap(self, parent_start: int, parent_end: int, child_start: int, child_end: int) -> bool:
        """
        Check if two chunks overlap in position.

        Args:
            parent_start: Start position of parent chunk
            parent_end: End position of parent chunk
            child_start: Start position of child chunk
            child_end: End position of child chunk

        Returns:
            True if chunks overlap
        """
        return (child_start >= parent_start and child_start <= parent_end) or \
               (child_end >= parent_start and child_end <= parent_end) or \
               (child_start <= parent_start and child_end >= parent_end)

    def _merge_chunks(self, candidates, top_k: int) -> List[Dict[str, Any]]:
        """Apply auto-merging logic to retrieved chunks."""
        merged_results = []

        documents = candidates['documents'][0]
        metadatas = candidates['metadatas'][0]
        distances = candidates['distances'][0]

        # Group chunks by document and level
        doc_chunks = defaultdict(lambda: defaultdict(list))

        for doc, metadata, distance in zip(documents, metadatas, distances):
            filename = metadata['filename']
            level = metadata['level']

            doc_chunks[filename][level].append({
                'text': doc,
                'metadata': metadata,
                'similarity': 1 - distance,
                'level': level
            })

        # For each document, apply merging logic
        for filename, level_chunks in doc_chunks.items():
            doc_results = self._merge_document_chunks(level_chunks, top_k)
            merged_results.extend(doc_results)

        # Sort by similarity and return top_k
        merged_results.sort(key=lambda x: x['similarity'], reverse=True)
        return merged_results[:top_k]

    def _merge_document_chunks(self, level_chunks: Dict[int, List], top_k: int) -> List[Dict[str, Any]]:
        """Merge chunks within a document using hierarchical logic."""
        results = []

        # Get leaf level chunks (smallest chunks)
        leaf_level = max(level_chunks.keys())
        leaf_chunks = level_chunks[leaf_level]

        # Sort by similarity
        leaf_chunks.sort(key=lambda x: x['similarity'], reverse=True)

        # For top chunks, try to merge with parents
        for chunk in leaf_chunks[:top_k]:
            merged_chunk = self._get_merged_content(chunk, level_chunks)
            results.append(merged_chunk)

        return results

    def _get_merged_content(self, chunk: Dict[str, Any], level_chunks: Dict[int, List]) -> Dict[str, Any]:
        """Get merged content for a chunk, preferring larger parent chunks when beneficial."""
        current_level = chunk['level']
        parent_id = chunk['metadata'].get('parent_id')

        # Check if parent chunk should be used instead
        if self._should_use_parent_chunk(chunk, level_chunks):
            parent_chunk = self._find_parent_chunk(chunk, level_chunks)
            if parent_chunk:
                return self._create_chunk_result(parent_chunk)

        # Return the original chunk
        return self._create_chunk_result(chunk)

    def _should_use_parent_chunk(self, chunk: Dict[str, Any], level_chunks: Dict[int, List]) -> bool:
        """
        Determine if a parent chunk should be used instead of the current chunk.

        Args:
            chunk: Current chunk dictionary
            level_chunks: Dictionary of chunks by level

        Returns:
            True if parent chunk should be used
        """
        parent_id = chunk['metadata'].get('parent_id')
        current_level = chunk['level']

        return parent_id and current_level > 0 and (current_level - 1) in level_chunks

    def _find_parent_chunk(self, chunk: Dict[str, Any], level_chunks: Dict[int, List]) -> Optional[Dict[str, Any]]:
        """
        Find the parent chunk for a given chunk.

        Args:
            chunk: Current chunk dictionary
            level_chunks: Dictionary of chunks by level

        Returns:
            Parent chunk if found and better, None otherwise
        """
        parent_id = chunk['metadata'].get('parent_id')
        parent_level = chunk['level'] - 1

        for parent_chunk in level_chunks[parent_level]:
            if parent_chunk['metadata'].get('id') == parent_id:
                # Check if parent has significantly higher similarity
                parent_similarity = parent_chunk['similarity']
                child_similarity = chunk['similarity']

                if parent_similarity > child_similarity + getattr(self, 'parent_similarity_threshold', 0.1):
                    return parent_chunk

        return None

    def _create_chunk_result(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized result dictionary for a chunk.

        Args:
            chunk: Chunk dictionary

        Returns:
            Standardized result dictionary
        """
        return {
            'document': chunk['text'],
            'metadata': chunk['metadata'],
            'similarity': chunk['similarity']
        }

    def get_loaded_documents(self) -> List[str]:
        """Get list of loaded document filenames."""
        return list(self.loaded_documents)

    def clear_documents(self):
        """Clear all loaded documents."""
        try:
            # Delete collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.loaded_documents.clear()
            logger.info("Cleared all auto-merging documents")
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
