"""Hierarchical chunking that preserves document structure.

Chunks documents while respecting structural boundaries (articles, sections, paragraphs)
and preserving hierarchy metadata for better context preservation.
"""

from typing import Dict, List, Optional

from cubo.ingestion.structure_detector import StructureNode, detect_document_structure
from cubo.monitoring import metrics
from cubo.utils.logger import logger
from cubo.utils.utils import Utils


class HierarchicalChunker:
    """Chunk documents while preserving structural boundaries."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        max_chunk_tokens: Optional[int] = None,
        min_chunk_tokens: Optional[int] = None,
        tokenizer_name: Optional[str] = None,
        preserve_boundaries: bool = True,
        include_parent_context: bool = True,
        overlap_sentences: int = 1,
        max_overlap_tokens: Optional[int] = None,
    ):
        """Initialize hierarchical chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
            max_chunk_tokens: Maximum chunk size in tokens (optional)
            min_chunk_tokens: Minimum chunk size in tokens (optional)
            tokenizer_name: Name of tokenizer to use for counting (optional)
            preserve_boundaries: Don't split structural units (articles, sections)
            include_parent_context: Add parent section titles to metadata
            overlap_sentences: Number of sentences to overlap between chunks
            max_overlap_tokens: Optional cap on overlapping tokens to control redundancy
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.preserve_boundaries = preserve_boundaries
        self.include_parent_context = include_parent_context
        self.overlap_sentences = overlap_sentences
        self.max_overlap_tokens = max_overlap_tokens

        self.tokenizer = None
        if tokenizer_name:
            try:
                import os
                from pathlib import Path

                from transformers import AutoTokenizer

                # If a local path is provided, load directly (safe).
                if Path(tokenizer_name).exists():
                    # Local load - safe to call without revision
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name, use_fast=True
                    )  # nosec
                else:
                    # Remote HF repo - require pinned revision or explicit opt-in
                    rev = os.getenv("HF_PINNED_REVISION")
                    allow_unpinned = os.getenv("HF_ALLOW_UNPINNED_HF_DOWNLOADS", "0") == "1"
                    if rev:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_name, revision=rev, use_fast=True
                        )  # nosec
                    elif allow_unpinned:
                        logger.warning(
                            f"Loading tokenizer {tokenizer_name} without pinned revision because HF_ALLOW_UNPINNED_HF_DOWNLOADS=1."
                        )
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_name, use_fast=True
                        )  # nosec
                    else:
                        raise RuntimeError(
                            "Attempted to download tokenizer without pinned HF revision. Set HF_PINNED_REVISION or HF_ALLOW_UNPINNED_HF_DOWNLOADS=1 to proceed."
                        )
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")

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
        import time

        start_time = time.time()

        # Detect structure if not provided
        if structure is None:
            structure = detect_document_structure(text, format_type)

        # If no structure detected, fall back to simple chunking
        if not structure:
            chunks = self._simple_chunk(text)
        else:
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

        elapsed = time.time() - start_time
        logger.info(f"Chunking completed in {elapsed:.4f}s. Generated {len(chunks)} chunks.")
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
            "token_count": Utils._token_count(text, self.tokenizer),
        }

        # Metrics & logging
        try:
            metrics.record("chunks_created", 1)
            if chunk["token_count"] is not None:
                metrics.record("chunk_tokens_total", chunk["token_count"])

            if (
                self.max_chunk_tokens is not None
                and chunk["token_count"] is not None
                and chunk["token_count"] >= 0.9 * self.max_chunk_tokens
            ):
                logger.warning(
                    f"Chunk {chunk_index} is approaching token limit: {chunk['token_count']} tokens (limit {self.max_chunk_tokens})"
                )

            if self.max_chunk_size is not None and len(text) >= 0.9 * self.max_chunk_size:
                logger.warning(
                    f"Chunk {chunk_index} is approaching char size limit: {len(text)} chars (limit {self.max_chunk_size})"
                )
        except Exception as e:
            logger.debug(f"Metrics recording failed: {e}")

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
        import re

        chunks = []
        # raise Exception("I am running updated code")

        # Split by paragraphs (double newline) to preserve structure
        # Use robust regex for paragraph breaks (handles \r\n, multiple newlines, spaces in between)
        paragraphs = re.split(r"\n\s*\n", text)
        # print(f"DEBUG: Paragraphs split: {len(paragraphs)}")
        if len(paragraphs) > 1:
            logger.debug(f"Splitting node into {len(paragraphs)} paragraphs")

        current_chunk = []
        current_size = 0
        current_tokens = 0

        def save_chunk():
            nonlocal current_chunk, current_size, current_tokens
            if not current_chunk:
                return

            chunk_text = " ".join(current_chunk)
            chunk = self._create_chunk(
                text=chunk_text,
                node=node,
                hierarchy=hierarchy,
                chunk_index=start_index + len(chunks),
            )
            chunks.append(chunk)
            logger.debug(
                f"Created chunk {len(chunks)}: {len(chunk_text)} chars, {chunk['token_count']} tokens"
            )

            # Start new chunk with overlap
            if self.overlap_sentences > 0:
                current_chunk = current_chunk[-self.overlap_sentences :]
                current_size = sum(len(s) for s in current_chunk)
                current_tokens = sum(Utils._token_count(s, self.tokenizer) for s in current_chunk)

                if self.max_overlap_tokens is not None and current_tokens > self.max_overlap_tokens:
                    logger.warning(
                        f"Overlap tokens ({current_tokens}) exceed max_overlap_tokens ({self.max_overlap_tokens})"
                    )
            else:
                current_chunk = []
                current_size = 0
                current_tokens = 0

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Split paragraph into sentences
            sentences = Utils._split_into_sentences(paragraph)

            # Pre-calculate sentence metrics
            p_data = []
            p_size = 0
            p_tokens = 0
            for s in sentences:
                length = len(s)
                t = Utils._token_count(s, self.tokenizer)
                p_data.append((s, length, t))
                p_size += length
                p_tokens += t

            # Check if adding whole paragraph exceeds limits
            size_exceeded = current_size + p_size > self.max_chunk_size
            tokens_exceeded = (self.max_chunk_tokens is not None) and (
                current_tokens + p_tokens > self.max_chunk_tokens
            )

            # print(f"DEBUG: P='{paragraph[:10]}...', size={p_size}, current={current_size}, max={self.max_chunk_size}, exceeded={size_exceeded}")

            if size_exceeded or tokens_exceeded:
                # If we have content, save it to respect paragraph boundary
                if current_chunk:
                    logger.debug("Paragraph boundary triggered chunk split")
                    save_chunk()

                # Check if paragraph fits in new chunk (which might have overlap)
                size_exceeded = current_size + p_size > self.max_chunk_size
                tokens_exceeded = (self.max_chunk_tokens is not None) and (
                    current_tokens + p_tokens > self.max_chunk_tokens
                )

                if not (size_exceeded or tokens_exceeded):
                    # Fits now (after clearing previous content)
                    for s, length, t in p_data:
                        current_chunk.append(s)
                        current_size += length
                        current_tokens += t
                else:
                    # Paragraph too big, must split internally
                    logger.debug("Paragraph too large, splitting internally")
                    for s, length, t in p_data:
                        # Check limits for single sentence addition
                        if (current_size + length > self.max_chunk_size) or (
                            (self.max_chunk_tokens is not None)
                            and (current_tokens + t > self.max_chunk_tokens)
                        ):
                            if current_chunk:
                                save_chunk()

                        if (length > self.max_chunk_size) or (
                            (self.max_chunk_tokens is not None) and (t > self.max_chunk_tokens)
                        ):
                            logger.warning(
                                f"Single sentence exceeds limits: {length} chars, {t} tokens"
                            )

                        current_chunk.append(s)
                        current_size += length
                        current_tokens += t
            else:
                # Fits, add all
                for s, length, t in p_data:
                    current_chunk.append(s)
                    current_size += length
                    current_tokens += t

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            # Always add if it's the only chunk, otherwise check min size
            size_ok = len(chunk_text) >= self.min_chunk_size
            tokens_ok = (self.min_chunk_tokens is None) or (
                Utils._token_count(chunk_text, self.tokenizer) >= self.min_chunk_tokens
            )

            if len(chunks) == 0 or (size_ok and tokens_ok):
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

    def _simple_chunk(self, text: str) -> List[Dict]:
        """Fallback to simple sentence-based chunking.

        Args:
            text: Text to chunk

        Returns:
            List of chunk dictionaries
        """
        import re

        chunks = []

        # Split by paragraphs (double newline) to preserve structure
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            logger.debug(f"Simple chunking: splitting into {len(paragraphs)} paragraphs")

        current_chunk = []
        current_size = 0
        current_tokens = 0

        def save_chunk():
            nonlocal current_chunk, current_size, current_tokens
            if not current_chunk:
                return

            chunk_text = " ".join(current_chunk)

            token_count = Utils._token_count(chunk_text, self.tokenizer)
            chunks.append(
                {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "type": "text",
                    "hierarchy": [],
                    "hierarchy_level": 0,
                    "parent_context": "",
                    "token_count": token_count,
                }
            )

            # Metrics & warnings for simple chunk
            try:
                metrics.record("chunks_created", 1)
                (
                    metrics.record("chunk_tokens_total", token_count)
                    if token_count is not None
                    else None
                )
                if (
                    self.max_chunk_tokens is not None
                    and token_count is not None
                    and token_count >= 0.9 * self.max_chunk_tokens
                ):
                    logger.warning(
                        f"Simple chunk is approaching token limit: {token_count} tokens (limit {self.max_chunk_tokens})"
                    )
            except Exception as e:
                logger.debug(f"Metrics recording failed: {e}")

            logger.debug(f"Created simple chunk {len(chunks)}: {len(chunk_text)} chars")

            # Overlap
            if self.overlap_sentences > 0:
                current_chunk = current_chunk[-self.overlap_sentences :]
                current_size = sum(len(s) for s in current_chunk)
                current_tokens = sum(Utils._token_count(s, self.tokenizer) for s in current_chunk)

                if self.max_overlap_tokens is not None and current_tokens > self.max_overlap_tokens:
                    logger.warning(
                        f"Overlap tokens ({current_tokens}) exceed max_overlap_tokens ({self.max_overlap_tokens})"
                    )
            else:
                current_chunk = []
                current_size = 0
                current_tokens = 0

        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # Split paragraph into sentences
            sentences = Utils._split_into_sentences(paragraph)

            # Pre-calculate sentence metrics
            p_data = []
            p_size = 0
            p_tokens = 0
            for s in sentences:
                length = len(s)
                t = Utils._token_count(s, self.tokenizer)
                p_data.append((s, length, t))
                p_size += length
                p_tokens += t

            # Check if adding whole paragraph exceeds limits
            size_exceeded = current_size + p_size > self.max_chunk_size
            tokens_exceeded = (self.max_chunk_tokens is not None) and (
                current_tokens + p_tokens > self.max_chunk_tokens
            )

            if size_exceeded or tokens_exceeded:
                # If we have content, save it to respect paragraph boundary
                if current_chunk:
                    logger.debug("Paragraph boundary triggered chunk split")
                    save_chunk()

                # Check if paragraph fits in new chunk (which might have overlap)
                size_exceeded = current_size + p_size > self.max_chunk_size
                tokens_exceeded = (self.max_chunk_tokens is not None) and (
                    current_tokens + p_tokens > self.max_chunk_tokens
                )

                if not (size_exceeded or tokens_exceeded):
                    # Fits now (after clearing previous content)
                    for s, length, t in p_data:
                        current_chunk.append(s)
                        current_size += length
                        current_tokens += t
                else:
                    # Paragraph too big, must split internally
                    logger.debug("Paragraph too large, splitting internally")
                    for s, length, t in p_data:
                        # Check limits for single sentence addition
                        if (current_size + length > self.max_chunk_size) or (
                            (self.max_chunk_tokens is not None)
                            and (current_tokens + t > self.max_chunk_tokens)
                        ):
                            if current_chunk:
                                save_chunk()

                        if (length > self.max_chunk_size) or (
                            (self.max_chunk_tokens is not None) and (t > self.max_chunk_tokens)
                        ):
                            logger.warning(
                                f"Single sentence exceeds limits: {length} chars, {t} tokens"
                            )

                        current_chunk.append(s)
                        current_size += length
                        current_tokens += t
            else:
                # Fits, add all
                for s, length, t in p_data:
                    current_chunk.append(s)
                    current_size += length
                    current_tokens += t

        # Add remaining
        if current_chunk:
            chunk_text = " ".join(current_chunk)

            size_ok = len(chunk_text) >= self.min_chunk_size
            tokens_ok = (self.min_chunk_tokens is None) or (
                Utils._token_count(chunk_text, self.tokenizer) >= self.min_chunk_tokens
            )

            if len(chunks) == 0 or (size_ok and tokens_ok):
                token_count = Utils._token_count(chunk_text, self.tokenizer)
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": len(chunks),
                        "type": "text",
                        "hierarchy": [],
                        "hierarchy_level": 0,
                        "parent_context": "",
                        "token_count": token_count,
                    }
                )

                try:
                    metrics.record("chunks_created", 1)
                    (
                        metrics.record("chunk_tokens_total", token_count)
                        if token_count is not None
                        else None
                    )
                except Exception as e:
                    logger.debug(f"Metrics recording failed: {e}")

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
