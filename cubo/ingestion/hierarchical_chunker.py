"""Hierarchical chunking that preserves document structure.

Chunks documents while respecting structural boundaries (articles, sections, paragraphs)
and preserving hierarchy metadata for better context preservation.
"""

from typing import List, Dict, Optional, Tuple
from cubo.ingestion.structure_detector import StructureNode, detect_document_structure


class HierarchicalChunker:
    """Chunk documents while preserving structural boundaries."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        preserve_boundaries: bool = True,
        include_parent_context: bool = True,
        overlap_sentences: int = 1,
    ):
        """Initialize hierarchical chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
            preserve_boundaries: Don't split structural units (articles, sections)
            include_parent_context: Add parent section titles to metadata
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_boundaries = preserve_boundaries
        self.include_parent_context = include_parent_context
        self.overlap_sentences = overlap_sentences

    def chunk(
        self, text: str, structure: Optional[List[StructureNode]] = None, format_type: str = "auto"
    ) -> List[Dict]:
        """Chunk text while preserving structure.

        Args:
            text: Document text to chunk
            structure: Pre-detected structure nodes (optional)
            format_type: Document format for structure detection

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Detect structure if not provided
        if structure is None:
            structure = detect_document_structure(text, format_type)

        # If no structure detected, fall back to simple chunking
        if not structure:
            return self._simple_chunk(text)

        # Chunk based on structure
        chunks = []
        hierarchy_stack = []  # Track current hierarchy path

        for i, node in enumerate(structure):
            # Update hierarchy stack
            hierarchy_stack = self._update_hierarchy(hierarchy_stack, node)

            # Extract text for this node
            node_text = text[node.start : node.end].strip()

            # If node is small enough, create single chunk
            if len(node_text) <= self.max_chunk_size:
                chunk = self._create_chunk(
                    text=node_text,
                    node=node,
                    hierarchy=hierarchy_stack.copy(),
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)
            else:
                # Split large nodes into sub-chunks
                sub_chunks = self._split_large_node(
                    text=node_text,
                    node=node,
                    hierarchy=hierarchy_stack.copy(),
                    start_index=len(chunks),
                )
                chunks.extend(sub_chunks)

        return chunks

    def _update_hierarchy(
        self, stack: List[StructureNode], node: StructureNode
    ) -> List[StructureNode]:
        """Update hierarchy stack with new node.

        Args:
            stack: Current hierarchy stack
            node: New node to add

        Returns:
            Updated hierarchy stack
        """
        # Remove nodes at same or lower level
        while stack and stack[-1].level >= node.level:
            stack.pop()

        # Add new node
        stack.append(node)
        return stack

    def _create_chunk(
        self, text: str, node: StructureNode, hierarchy: List[StructureNode], chunk_index: int
    ) -> Dict:
        """Create chunk dictionary with metadata.

        Args:
            text: Chunk text
            node: Structure node for this chunk
            hierarchy: Hierarchy path to this chunk
            chunk_index: Index of this chunk

        Returns:
            Chunk dictionary with text and metadata
        """
        # Build hierarchy path (titles)
        hierarchy_path = [n.title for n in hierarchy]

        # Build parent context (for better retrieval)
        parent_context = ""
        if self.include_parent_context and len(hierarchy) > 1:
            parent_context = " → ".join(hierarchy_path[:-1])

        chunk = {
            "text": text,
            "chunk_index": chunk_index,
            "type": node.type,
            "hierarchy": hierarchy_path,
            "hierarchy_level": node.level,
            "parent_context": parent_context,
            "section_title": node.title,
            "numbering": node.numbering,
            "token_count": len(text.split()),
        }

        return chunk

    def _split_large_node(
        self, text: str, node: StructureNode, hierarchy: List[StructureNode], start_index: int
    ) -> List[Dict]:
        """Split large node into multiple chunks.

        Args:
            text: Node text to split
            node: Structure node
            hierarchy: Hierarchy path
            start_index: Starting chunk index

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        # Split by sentences
        sentences = self._split_sentences(text)

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds max size, create chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = self._create_chunk(
                    text=chunk_text,
                    node=node,
                    hierarchy=hierarchy,
                    chunk_index=start_index + len(chunks),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap_sentences > 0:
                    current_chunk = current_chunk[-self.overlap_sentences :]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            # Always add if it's the only chunk, otherwise check min size
            if len(chunks) == 0 or len(chunk_text) >= self.min_chunk_size:
                chunk = self._create_chunk(
                    text=chunk_text,
                    node=node,
                    hierarchy=hierarchy,
                    chunk_index=start_index + len(chunks),
                )
                chunks.append(chunk)

        # Ensure we always return at least one chunk
        if not chunks and text.strip():
            chunk = self._create_chunk(
                text=text, node=node, hierarchy=hierarchy, chunk_index=start_index
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK)
        import re

        # Split on sentence boundaries
        sentences = re.split(r"([.!?]+\s+)", text)

        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            result.append(sentence.strip())

        # Add last sentence if odd number
        if len(sentences) % 2 == 1:
            result.append(sentences[-1].strip())

        return [s for s in result if s]

    def _simple_chunk(self, text: str) -> List[Dict]:
        """Fallback to simple sentence-based chunking.

        Args:
            text: Text to chunk

        Returns:
            List of chunk dictionaries
        """
        sentences = self._split_sentences(text)
        chunks = []

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": len(chunks),
                        "type": "text",
                        "hierarchy": [],
                        "hierarchy_level": 0,
                        "parent_context": "",
                        "token_count": len(chunk_text.split()),
                    }
                )

                # Overlap
                if self.overlap_sentences > 0:
                    current_chunk = current_chunk[-self.overlap_sentences :]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "type": "text",
                    "hierarchy": [],
                    "hierarchy_level": 0,
                    "parent_context": "",
                    "token_count": len(chunk_text.split()),
                }
            )

        return chunks


def chunk_with_structure(
    text: str, max_chunk_size: int = 1000, format_type: str = "auto"
) -> List[Dict]:
    """Convenience function for hierarchical chunking.

    Args:
        text: Document text
        max_chunk_size: Maximum chunk size
        format_type: Document format

    Returns:
        List of chunk dictionaries
    """
    chunker = HierarchicalChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk(text, format_type=format_type)
