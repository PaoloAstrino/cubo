"""Unit tests for structure detector."""

import pytest

from cubo.ingestion.structure_detector import (
    LegalDocumentDetector,
    MarkdownStructureDetector,
    StructureNode,
    TextStructureDetector,
    detect_document_structure,
)


class TestStructureNode:
    """Test StructureNode class."""

    def test_node_creation(self):
        """Test creating a structure node."""
        node = StructureNode(
            level=1, node_type="h1", title="Introduction", start_pos=0, end_pos=100, numbering="1"
        )

        assert node.level == 1
        assert node.type == "h1"
        assert node.title == "Introduction"
        assert node.numbering == "1"

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = StructureNode(1, "h1", "Test", 0, 100)
        d = node.to_dict()

        assert d["level"] == 1
        assert d["type"] == "h1"
        assert d["title"] == "Test"


class TestMarkdownStructureDetector:
    """Test Markdown structure detection."""

    def test_detect_headers(self):
        """Test detecting Markdown headers."""
        detector = MarkdownStructureDetector()

        markdown = """# Article 5
## Paragraph 1
Some text here.
## Paragraph 2
More text.
### Subparagraph
Details."""

        nodes = detector.detect(markdown)

        assert len(nodes) == 4
        assert nodes[0].level == 1
        assert nodes[0].title == "Article 5"
        assert nodes[1].level == 2
        assert nodes[1].title == "Paragraph 1"

    def test_empty_document(self):
        """Test empty document."""
        detector = MarkdownStructureDetector()
        nodes = detector.detect("")
        assert len(nodes) == 0

    def test_no_headers(self):
        """Test document with no headers."""
        detector = MarkdownStructureDetector()
        nodes = detector.detect("Just plain text without headers.")
        assert len(nodes) == 0


class TestLegalDocumentDetector:
    """Test legal document structure detection."""

    def test_detect_articles(self):
        """Test detecting articles."""
        detector = LegalDocumentDetector()

        legal_text = """Article 5: Data Protection Rights
The data subject shall have the right to access.

Article 6: Lawful Processing
Processing shall be lawful only if..."""

        nodes = detector.detect(legal_text)

        # Should detect both articles
        articles = [n for n in nodes if n.type == "article"]
        assert len(articles) >= 2
        assert articles[0].numbering == "5"
        assert "Data Protection Rights" in articles[0].title

    def test_detect_sections(self):
        """Test detecting sections."""
        detector = LegalDocumentDetector()

        legal_text = """Section 1: Introduction
This section introduces the topic.

Section 2: Definitions
Terms are defined here."""

        nodes = detector.detect(legal_text)

        sections = [n for n in nodes if n.type == "section"]
        assert len(sections) >= 2
        assert sections[0].numbering == "1"

    def test_detect_paragraphs(self):
        """Test detecting paragraphs."""
        detector = LegalDocumentDetector()

        legal_text = """§ 1. First paragraph
Content of first paragraph.

§ 2. Second paragraph
Content of second paragraph."""

        nodes = detector.detect(legal_text)

        paragraphs = [n for n in nodes if n.type == "paragraph"]
        assert len(paragraphs) >= 2
        assert paragraphs[0].numbering.strip(".") == "1"  # Accept '1' or '1.'

    def test_mixed_structure(self):
        """Test document with mixed structure."""
        detector = LegalDocumentDetector()

        legal_text = """Article 5: Data Rights
Section 1: Access Rights
§ 1. The data subject has the right to access.
§ 2. Access shall be provided within 30 days.

Section 2: Rectification
§ 1. The data subject may request rectification."""

        nodes = detector.detect(legal_text)

        # Should detect all levels
        assert any(n.type == "article" for n in nodes)
        assert any(n.type == "section" for n in nodes)
        assert any(n.type == "paragraph" for n in nodes)


class TestTextStructureDetector:
    """Test plain text structure detection."""

    def test_detect_all_caps_headers(self):
        """Test detecting ALL CAPS headers."""
        detector = TextStructureDetector()

        text = """INTRODUCTION
This is the introduction section.

METHODOLOGY
This section describes the methodology."""

        nodes = detector.detect(text)

        assert len(nodes) >= 2
        assert "INTRODUCTION" in nodes[0].title

    def test_detect_numbered_sections(self):
        """Test detecting numbered sections."""
        detector = TextStructureDetector()

        text = """1. First Section
Content of first section.

2. Second Section
Content of second section."""

        nodes = detector.detect(text)

        assert len(nodes) >= 2


class TestAutoDetection:
    """Test automatic format detection."""

    def test_auto_detect_markdown(self):
        """Test auto-detecting Markdown."""
        text = "# Header\nSome content"
        nodes = detect_document_structure(text, format_type="auto")

        # Should detect as markdown
        assert len(nodes) > 0

    def test_auto_detect_legal(self):
        """Test auto-detecting legal document."""
        text = "Article 5: Data Rights\nSome content"
        nodes = detect_document_structure(text, format_type="auto")

        # Should detect as legal
        assert len(nodes) > 0
        assert any(n.type == "article" for n in nodes)

    def test_fallback_to_text(self):
        """Test fallback to text detection."""
        text = "Just plain text without structure."
        nodes = detect_document_structure(text, format_type="auto")

        # Should return empty or minimal structure
        assert isinstance(nodes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
