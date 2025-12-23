import pytest

from cubo.retrieval.bm25_python_store import BM25PythonStore
from cubo.utils.lemmatizer import SimplemmaLemmatizer


def test_lemmatizer_basic():
    lemmatizer = SimplemmaLemmatizer()
    if not lemmatizer.is_available():
        pytest.skip("simplemma not installed")

    # Italian
    assert lemmatizer.lemmatize_token("gatti", lang="it") == "gatto"
    # simplemma might return 'mangiare' or 'mangiato' depending on context/version,
    # but usually it's good with verbs.
    # Let's check a sure one.
    assert lemmatizer.lemmatize_token("mangiato", lang="it") == "mangiare"

    # English
    assert lemmatizer.lemmatize_token("cats", lang="en") == "cat"
    assert lemmatizer.lemmatize_token("running", lang="en") == "run"


def test_bm25_with_lemmatization():
    store = BM25PythonStore(use_lemmatization=True)
    if not store.lemmatizer or not store.lemmatizer.is_available():
        pytest.skip("simplemma not installed")

    # Index documents with inflected forms
    docs = [
        {"doc_id": "1", "text": "I gatti corrono veloci"},
        {"doc_id": "2", "text": "Il cane dorme"},
    ]
    store.index_documents(docs)

    # Search with base form
    # "gatto" should match "gatti" because "gatti" -> "gatto" and query "gatto" -> "gatto"
    results = store.search("gatto")
    assert len(results) > 0
    assert results[0]["doc_id"] == "1"

    # Search with infinitive
    # "corrono" -> "correre"
    results = store.search("correre")
    assert len(results) > 0
    assert results[0]["doc_id"] == "1"


def test_bm25_without_lemmatization_comparison():
    # Standard stemming (Snowball)
    store_stem = BM25PythonStore(use_lemmatization=False)

    # Lemmatization
    store_lemma = BM25PythonStore(use_lemmatization=True)
    if not store_lemma.lemmatizer or not store_lemma.lemmatizer.is_available():
        pytest.skip("simplemma not installed")

    text = "I gatti mangiano il cibo"

    # Force language to ensure we compare apples to apples
    tokens_stem = store_stem._tokenize(text, language="it")
    tokens_lemma = store_lemma._tokenize(text, language="it")

    # Snowball: gatti -> gatt, mangiano -> mang
    # Simplemma: gatti -> gatto, mangiano -> mangiare

    # Note: MultilingualTokenizer might return different token lists (e.g. stop words removal)
    # simplemma.text_lemmatizer does not remove stop words by default unless we filter them.
    # MultilingualTokenizer removes stop words if configured (default False in BM25PythonStore init).

    # Let's check specific tokens
    print(f"Stemmed: {tokens_stem}")
    print(f"Lemmatized: {tokens_lemma}")

    assert "gatt" in tokens_stem
    assert "gatto" in tokens_lemma
