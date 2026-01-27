"""Tests for German compound word splitting functionality."""

import pytest

from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer


def test_compound_splitter_disabled_by_default():
    """Test compound splitter is opt-in."""
    tokenizer = MultilingualTokenizer()
    assert not tokenizer.use_compound_splitter


def test_compound_splitter_enabled():
    """Test compound splitter can be enabled."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True)
    assert tokenizer.use_compound_splitter


def test_german_compound_splitting():
    """Test German compound words are split."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    # Test compound: "Datenschutzgrundverordnung" (GDPR in German)
    # Should split into: Datenschutz + grund + verordnung (or similar)
    tokens = tokenizer.tokenize("Datenschutzgrundverordnung", language="de")

    # Original word should be preserved
    assert any("datenschutzgrundverordnung" in t.lower() for t in tokens)

    # Should have more than just the original (some splits)
    assert len(tokens) > 1, f"Expected compound splitting, got: {tokens}"


def test_short_words_not_split():
    """Test short words are not split."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    tokens = tokenizer.tokenize("haus", language="de")  # "house" in German

    # Short word should not be split
    assert tokens == ["haus"]


def test_compound_splitting_increases_token_overlap():
    """Test compound splitting increases token overlap for retrieval."""
    tokenizer_no_split = MultilingualTokenizer(use_compound_splitter=False, use_stemming=False)
    tokenizer_with_split = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    # Query with sub-components of compound
    query = "Datenschutz Verordnung"  # "Data protection regulation"
    document = "Die Datenschutzgrundverordnung regelt..."  # "The GDPR regulates..."

    query_tokens_no_split = set(tokenizer_no_split.tokenize(query, language="de"))
    query_tokens_split = set(tokenizer_with_split.tokenize(query, language="de"))

    doc_tokens_no_split = set(tokenizer_no_split.tokenize(document, language="de"))
    doc_tokens_split = set(tokenizer_with_split.tokenize(document, language="de"))

    # Calculate overlap
    overlap_no_split = len(query_tokens_no_split & doc_tokens_no_split)
    overlap_split = len(query_tokens_split & doc_tokens_split)

    # With splitting, should have more overlap
    assert (
        overlap_split > overlap_no_split
    ), f"Expected more overlap with splitting. No split: {overlap_no_split}, Split: {overlap_split}"


def test_compound_splitting_preserves_original():
    """Test original compound word is preserved alongside splits."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    tokens = tokenizer.tokenize("Bundestagswahl", language="de")  # "Federal election"

    # Should contain both original and splits
    original_present = any("bundestagswahl" in t.lower() for t in tokens)
    has_multiple_tokens = len(tokens) > 1

    assert original_present, "Original compound word should be preserved"
    assert has_multiple_tokens, "Should have split forms in addition to original"


def test_non_german_languages_unaffected():
    """Test compound splitting only affects German."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    # English should not be affected
    tokens_en = tokenizer.tokenize("understanding", language="en")
    assert tokens_en == ["understanding"]

    # Italian should not be affected
    tokens_it = tokenizer.tokenize("comprensione", language="it")
    assert tokens_it == ["comprensione"]


@pytest.mark.parametrize(
    "compound,expected_parts",
    [
        ("Donaudampfschiff", ["donau", "dampf", "schiff"]),  # Danube steamship
        ("Arbeitszeitverordnung", ["arbeitszeit", "verordnung"]),  # Working hours regulation
    ],
)
def test_specific_german_compounds(compound, expected_parts):
    """Test specific German compound words split correctly."""
    tokenizer = MultilingualTokenizer(use_compound_splitter=True, use_stemming=False)

    tokens = tokenizer.tokenize(compound, language="de")
    tokens_lower = [t.lower() for t in tokens]

    # Check that at least some expected parts appear
    parts_found = sum(1 for part in expected_parts if any(part in t for t in tokens_lower))

    assert (
        parts_found >= 1
    ), f"Expected to find parts {expected_parts} in {tokens}, but found {parts_found}"
