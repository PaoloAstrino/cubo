"""Document structure detection for hierarchical chunking.

Detects document structure across different formats (Markdown, PDF, plain text)
to enable structure-aware chunking that preserves semantic boundaries.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class StructureNode:
    """Represents a structural element in a document."""

    def __init__(
        self,
        level: int,
        node_type: str,
        title: str,
        start_pos: int,
        end_pos: int,
        numbering: Optional[str] = None,
    ):
        """Initialize structure node.

        Args:
            level: Hierarchy level (1=top, 2=subsection, etc.)
            node_type: Type of node ('h1', 'h2', 'article', 'paragraph', etc.)
            title: Title/heading text
            start_pos: Start position in document (char index)
            end_pos: End position in document (char index)
            numbering: Optional numbering (e.g., "5", "2.3", "§ 1")
        """
        self.level = level
        self.type = node_type
        self.title = title
        self.start = start_pos
        self.end = end_pos
        self.numbering = numbering

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "level": self.level,
            "type": self.type,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "numbering": self.numbering,
        }

    def __repr__(self) -> str:
        return f"StructureNode(level={self.level}, type={self.type}, title='{self.title}')"


class StructureDetector(ABC):
    """Base class for document structure detection."""

    @abstractmethod
    def detect(self, content: str) -> List[StructureNode]:
        """Detect structure in document content.

        Args:
            content: Document text content

        Returns:
            List of StructureNode objects representing document hierarchy
        """
        pass

    @staticmethod
    def for_format(format_type: str) -> "StructureDetector":
        """Factory method to get detector for specific format.

        Args:
            format_type: Format type ('markdown', 'pdf', 'legal', 'text')

        Returns:
            Appropriate StructureDetector instance
        """
        if format_type.lower() in ["md", "markdown"]:
            return MarkdownStructureDetector()
        elif format_type.lower() == "pdf":
            return PDFStructureDetector()
        elif format_type.lower() == "legal":
            return LegalDocumentDetector()
        else:
            return TextStructureDetector()


class MarkdownStructureDetector(StructureDetector):
    """Detect structure in Markdown documents."""

    # Markdown header patterns
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def detect(self, content: str) -> List[StructureNode]:
        """Detect Markdown headers and create structure.

        Args:
            content: Markdown text

        Returns:
            List of StructureNode objects for each header
        """
        nodes = []

        for match in self.HEADER_PATTERN.finditer(content):
            hashes = match.group(1)
            title = match.group(2).strip()
            level = len(hashes)
            start_pos = match.start()

            # Find end position (next header or end of document)
            next_match = self.HEADER_PATTERN.search(content, match.end())
            end_pos = next_match.start() if next_match else len(content)

            node = StructureNode(
                level=level,
                node_type=f"h{level}",
                title=title,
                start_pos=start_pos,
                end_pos=end_pos,
            )
            nodes.append(node)

        return nodes


class PDFStructureDetector(StructureDetector):
    """Detect structure in PDF documents using font sizes and positions."""

    def detect(self, content: str) -> List[StructureNode]:
        """Detect PDF structure (placeholder - requires pdfplumber integration).

        Args:
            content: PDF text content

        Returns:
            List of StructureNode objects
        """
        # For now, fall back to legal document detection
        # Full PDF structure detection would require pdfplumber font analysis
        legal_detector = LegalDocumentDetector()
        return legal_detector.detect(content)


class LegalDocumentDetector(StructureDetector):
    """Detect structure in legal documents (articles, sections, paragraphs)."""

    # Legal document patterns
    ARTICLE_PATTERN = re.compile(
        r"^(Article|Art\.|ARTICLE)\s+(\d+[\w\.]*)[\s:.-]+(.+?)$", re.MULTILINE | re.IGNORECASE
    )

    SECTION_PATTERN = re.compile(
        r"^(Section|Sec\.|SECTION|§)\s+(\d+[\w\.]*)[\s:.-]+(.+?)$", re.MULTILINE | re.IGNORECASE
    )

    PARAGRAPH_PATTERN = re.compile(
        r"^(Paragraph|Para\.|§)\s+(\d+[\w\.]*)[\s:.-]*(.*)$", re.MULTILINE | re.IGNORECASE
    )

    # Numbered list patterns (1., 1), (a), etc.)
    NUMBERED_LIST_PATTERN = re.compile(r"^(\d+|[a-z])[.)]\s+(.+?)$", re.MULTILINE)

    def detect(self, content: str) -> List[StructureNode]:
        """Detect legal document structure.

        Args:
            content: Legal document text

        Returns:
            List of StructureNode objects for articles, sections, paragraphs
        """
        nodes = []

        # Detect articles (level 1)
        for match in self.ARTICLE_PATTERN.finditer(content):
            numbering = match.group(2)
            title = match.group(3).strip()
            start_pos = match.start()

            # Find end (next article or end of document)
            next_match = self.ARTICLE_PATTERN.search(content, match.end())
            end_pos = next_match.start() if next_match else len(content)

            node = StructureNode(
                level=1,
                node_type="article",
                title=title,
                start_pos=start_pos,
                end_pos=end_pos,
                numbering=numbering,
            )
            nodes.append(node)

        # Detect sections (level 2)
        for match in self.SECTION_PATTERN.finditer(content):
            numbering = match.group(2)
            title = match.group(3).strip()
            start_pos = match.start()

            # Find end (next section/article or end of document)
            next_section = self.SECTION_PATTERN.search(content, match.end())
            next_article = self.ARTICLE_PATTERN.search(content, match.end())

            end_pos = len(content)
            if next_section and next_article:
                end_pos = min(next_section.start(), next_article.start())
            elif next_section:
                end_pos = next_section.start()
            elif next_article:
                end_pos = next_article.start()

            node = StructureNode(
                level=2,
                node_type="section",
                title=title,
                start_pos=start_pos,
                end_pos=end_pos,
                numbering=numbering,
            )
            nodes.append(node)

        # Detect paragraphs (level 3)
        for match in self.PARAGRAPH_PATTERN.finditer(content):
            numbering = match.group(2)
            title = match.group(3).strip() if match.group(3) else f"Paragraph {numbering}"
            start_pos = match.start()

            # Find end (next paragraph/section/article or end of document)
            end_pos = self._find_next_boundary(content, match.end())

            node = StructureNode(
                level=3,
                node_type="paragraph",
                title=title,
                start_pos=start_pos,
                end_pos=end_pos,
                numbering=numbering,
            )
            nodes.append(node)

        # Sort by start position
        nodes.sort(key=lambda n: n.start)

        return nodes

    def _find_next_boundary(self, content: str, start_pos: int) -> int:
        """Find next structural boundary after given position."""
        boundaries = []

        for pattern in [self.ARTICLE_PATTERN, self.SECTION_PATTERN, self.PARAGRAPH_PATTERN]:
            match = pattern.search(content, start_pos)
            if match:
                boundaries.append(match.start())

        return min(boundaries) if boundaries else len(content)


class TextStructureDetector(StructureDetector):
    """Detect structure in plain text documents using heuristics."""

    # Patterns for common section markers
    SECTION_MARKERS = [
        re.compile(r"^([A-Z][A-Z\s]{3,})\s*$", re.MULTILINE),  # ALL CAPS HEADERS
        re.compile(r"^(\d+\.\s+[A-Z].+?)$", re.MULTILINE),  # 1. Numbered sections
        re.compile(r"^([IVX]+\.\s+.+?)$", re.MULTILINE),  # Roman numerals
    ]

    def detect(self, content: str) -> List[StructureNode]:
        """Detect structure in plain text using heuristics.

        Args:
            content: Plain text content

        Returns:
            List of StructureNode objects
        """
        nodes = []

        # Try each pattern
        for i, pattern in enumerate(self.SECTION_MARKERS):
            for match in pattern.finditer(content):
                title = match.group(1).strip()
                start_pos = match.start()

                # Find end (next match or end of document)
                next_match = pattern.search(content, match.end())
                end_pos = next_match.start() if next_match else len(content)

                node = StructureNode(
                    level=i + 1,  # Different patterns = different levels
                    node_type="section",
                    title=title,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
                nodes.append(node)

        # Sort by start position and remove overlaps
        nodes.sort(key=lambda n: n.start)
        nodes = self._remove_overlaps(nodes)

        return nodes

    def _remove_overlaps(self, nodes: List[StructureNode]) -> List[StructureNode]:
        """Remove overlapping nodes, keeping higher-level ones."""
        if not nodes:
            return []

        result = [nodes[0]]
        for node in nodes[1:]:
            last = result[-1]
            # If this node starts after last one ends, keep it
            if node.start >= last.end:
                result.append(node)
            # If this node is higher level (lower number), replace last
            elif node.level < last.level:
                result[-1] = node

        return result


def detect_document_structure(content: str, format_type: str = "auto") -> List[StructureNode]:
    """Convenience function to detect document structure.

    Args:
        content: Document text content
        format_type: Format type ('markdown', 'pdf', 'legal', 'text', 'auto')

    Returns:
        List of StructureNode objects
    """
    if format_type == "auto":
        # Try to auto-detect format
        if content.startswith("#") or "\n#" in content:
            format_type = "markdown"
        elif "Article" in content or "Section" in content or "§" in content:
            format_type = "legal"
        else:
            format_type = "text"

    detector = StructureDetector.for_format(format_type)
    return detector.detect(content)
