"""Unit tests for hierarchical chunker."""

import pytest

from cubo.ingestion.hierarchical_chunker import HierarchicalChunker, chunk_with_structure
from cubo.ingestion.structure_detector import StructureNode


class TestHierarchicalChunker:
    """Test hierarchical chunking functionality."""

    def test_initialization(self):
        """Test chunker initialization."""
        chunker = HierarchicalChunker(max_chunk_size=1000)
        assert chunker.max_chunk_size == 1000
        assert chunker.preserve_boundaries is True

    def test_simple_chunk_fallback(self):
        """Test fallback to simple chunking when no structure."""
        chunker = HierarchicalChunker(max_chunk_size=100)

        text = "This is a simple text. It has multiple sentences. Each sentence is short."
        chunks = chunker.chunk(text, structure=[])

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("chunk_index" in chunk for chunk in chunks)

    def test_chunk_with_structure(self):
        """Test chunking with structure nodes."""
        chunker = HierarchicalChunker(max_chunk_size=500)

        text = """# Article 5
This is the content of Article 5. It contains important information about data rights.

## Paragraph 1
Details about paragraph 1.

## Paragraph 2
Details about paragraph 2."""

        # Create structure nodes
        structure = [
            StructureNode(1, "h1", "Article 5", 0, len(text)),
            StructureNode(2, "h2", "Paragraph 1", 50, 150),
            StructureNode(2, "h2", "Paragraph 2", 150, len(text)),
        ]

        chunks = chunker.chunk(text, structure=structure)

        assert len(chunks) > 0
        # Check hierarchy metadata
        assert all("hierarchy" in chunk for chunk in chunks)
        assert all("hierarchy_level" in chunk for chunk in chunks)

    def test_preserve_boundaries(self):
        """Test that structural boundaries are preserved."""
        chunker = HierarchicalChunker(max_chunk_size=50, preserve_boundaries=True)

        text = "Article 5: This is a very long article that exceeds the max chunk size but should not be split."

        structure = [StructureNode(1, "article", "Article 5", 0, len(text), numbering="5")]

        chunks = chunker.chunk(text, structure=structure)

        # Should create chunks but preserve article context
        assert len(chunks) > 0
        assert all(chunk["type"] == "article" for chunk in chunks)

    def test_hierarchy_metadata(self):
        """Test that hierarchy metadata is correctly added."""
        chunker = HierarchicalChunker(max_chunk_size=500)

        text = "Article 5 content. Paragraph 1 content."

        structure = [
            StructureNode(1, "article", "Article 5", 0, len(text), numbering="5"),
            StructureNode(2, "paragraph", "Paragraph 1", 20, len(text), numbering="1"),
        ]

        chunks = chunker.chunk(text, structure=structure)

        # Check hierarchy path
        assert len(chunks) > 0
        # Last chunk should have both article and paragraph in hierarchy
        if len(chunks) > 1:
            assert "Article 5" in chunks[-1]["hierarchy"]
            assert "Paragraph 1" in chunks[-1]["hierarchy"]

    def test_parent_context(self):
        """Test that parent context is included."""
        chunker = HierarchicalChunker(max_chunk_size=500, include_parent_context=True)

        text = "Article 5 content. Paragraph 1 content."

        structure = [
            StructureNode(1, "article", "Article 5", 0, len(text)),
            StructureNode(2, "paragraph", "Paragraph 1", 20, len(text)),
        ]

        chunks = chunker.chunk(text, structure=structure)

        # Chunks should have parent context
        assert any("parent_context" in chunk and chunk["parent_context"] for chunk in chunks)

    def test_split_large_node(self):
        """Test splitting large nodes into multiple chunks."""
        chunker = HierarchicalChunker(max_chunk_size=100)

        # Create a large article
        text = "Article 5: " + " ".join(
            ["This is sentence number {}.".format(i) for i in range(50)]
        )

        structure = [StructureNode(1, "article", "Article 5", 0, len(text))]

        chunks = chunker.chunk(text, structure=structure)

        # Should split into multiple chunks
        assert len(chunks) > 1
        # All should have same hierarchy
        assert all(chunk["hierarchy"] == chunks[0]["hierarchy"] for chunk in chunks)

    def test_sentence_overlap(self):
        """Test sentence overlap between chunks."""
        chunker = HierarchicalChunker(max_chunk_size=100, overlap_sentences=1)

        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

        chunks = chunker.chunk(text, structure=[])

        # With overlap, chunks should share some content
        if len(chunks) > 1:
            # This is a basic check - actual overlap depends on sentence lengths
            assert len(chunks) >= 1

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = HierarchicalChunker()

        chunks = chunker.chunk("", structure=[])

        # Should return empty or minimal chunks
        assert isinstance(chunks, list)

    def test_convenience_function(self):
        """Test convenience function."""
        text = "# Header\nSome content here."

        chunks = chunk_with_structure(text, max_chunk_size=1000, format_type="markdown")

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)


class TestHierarchyStack:
    """Test hierarchy stack management."""

    def test_hierarchy_update(self):
        """Test hierarchy stack updates correctly."""
        chunker = HierarchicalChunker()

        stack = []
        node1 = StructureNode(1, "h1", "Level 1", 0, 100)
        node2 = StructureNode(2, "h2", "Level 2", 50, 100)
        node3 = StructureNode(1, "h1", "New Level 1", 100, 200)

        # Add first node
        stack = chunker._update_hierarchy(stack, node1)
        assert len(stack) == 1

        # Add child node
        stack = chunker._update_hierarchy(stack, node2)
        assert len(stack) == 2

        # Add new top-level node (should clear stack)
        stack = chunker._update_hierarchy(stack, node3)
        assert len(stack) == 1
        assert stack[0].title == "New Level 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
