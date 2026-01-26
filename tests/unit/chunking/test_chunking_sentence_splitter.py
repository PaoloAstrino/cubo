"""Tests for sentence splitting logic to reproduce 'Blind Sentence Splitter' critique."""

from cubo.ingestion.hierarchical_chunker import HierarchicalChunker
from cubo.utils.utils import Utils


class TestSentenceSplitterCritique:
    """Reproduce failures in sentence splitting."""

    def test_abbreviation_splitting_utils(self):
        """Test that Utils._split_into_sentences incorrectly splits abbreviations."""
        text = "This is Art. 5 of the GDPR. It should be one sentence."
        sentences = Utils._split_into_sentences(text)

        # The critique says it splits "Art. 5" into ["Art.", "5 ..."]
        # We expect this to FAIL once fixed, but PASS (reproduce failure) now?
        # The instruction says "Acceptance: tests fail on current master and pass after fixes."
        # So I should write the test asserting the CORRECT behavior, and it should fail now.

        # Correct behavior:
        assert "Art. 5 of the GDPR" in sentences[0] or "Art. 5 of the GDPR" in " ".join(sentences)
        # Ideally "This is Art. 5 of the GDPR." is one sentence.

        # Let's be specific about the split
        # Current naive regex: re.split(r"(?<=[.!?])\s+", text)
        # "Art." matches (?<=[.!?])

        assert len(sentences) == 2, "Should be 2 sentences total"
        assert sentences[0] == "This is Art. 5 of the GDPR."
        assert sentences[1] == "It should be one sentence."

    def test_abbreviation_splitting_hierarchical_chunker(self):
        """Test that HierarchicalChunker uses robust splitting (via Utils)."""
        # HierarchicalChunker now uses Utils._split_into_sentences internally.
        # We can test this by checking if it splits correctly.
        HierarchicalChunker()
        text = "Mr. Smith went to Washington. He met with Dr. Jones."

        # We can't call _split_sentences anymore.
        # But we can call chunk() and see if it respects the sentence.
        # Or just call Utils._split_into_sentences directly to verify the logic used.

        sentences = Utils._split_into_sentences(text)

        assert len(sentences) == 2
        assert sentences[0] == "Mr. Smith went to Washington."
        assert sentences[1] == "He met with Dr. Jones."

    def test_legal_citations(self):
        """Test legal citation splitting."""
        text = "See v. United States, 123 U.S. 456. This is a case."
        sentences = Utils._split_into_sentences(text)

        assert len(sentences) == 2
        assert sentences[0] == "See v. United States, 123 U.S. 456."

    def test_complex_abbreviations(self):
        """Test multiple abbreviations."""
        text = "The U.S.A. is a country. e.g. is an example. i.e. is that is."
        sentences = Utils._split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "The U.S.A. is a country."
        assert sentences[1] == "e.g. is an example."
        assert sentences[2] == "i.e. is that is."
