import pytest
from cubo.retrieval.multilingual_tokenizer import MultilingualTokenizer, tokenize_multilingual
from cubo.retrieval.bm25_python_store import BM25PythonStore

def test_multilingual_stemming_italian():
    """Verify Italian stemming works (gatto/gatti match)."""
    tokenizer = MultilingualTokenizer()
    
    # Stemming check
    t1 = tokenizer.tokenize("Il gatto mangia", language="it")
    t2 = tokenizer.tokenize("I gatti mangiano", language="it")
    
    # "gatto" and "gatti" should stem to the same root (likely 'gatt')
    assert "gatt" in t1
    assert "gatt" in t2
    
    # Intersection should not be empty
    common = set(t1).intersection(set(t2))
    assert len(common) >= 1, f"Expected overlap between {t1} and {t2}"
    print("Italian stemming verified: gatto/gatti -> gatt")

def test_bm25_store_uses_advanced_tokenization():
    """Verify BM25PythonStore uses the new tokenizer and finds matches across forms."""
    store = BM25PythonStore(use_lemmatization=False) # Use stemming path
    
    # Add document with plural (longer text for reliable lang detection)
    text = "In Italia ci sono molti gatti che corrono veloci nei vicoli di Roma antica."
    docs = [{"doc_id": "1", "text": text}]
    store.add_documents(docs)
    
    # Search with singular
    results = store.search("gatto")
    
    # Debug info if failure
    if len(results) == 0:
        print(f"DEBUG: Doc tokens: {store._tokenize(text)}")
        print(f"DEBUG: Query tokens: {store._tokenize('gatto')}")
    
    assert len(results) > 0, "Failed to match 'gatto' with 'gatti' using BM25 store"
    assert results[0]["doc_id"] == "1"
    print("BM25 Store advanced tokenization verified.")

def test_language_detection():
    """Verify basic language detection."""
    tokenizer = MultilingualTokenizer()
    
    lang_it = tokenizer.detect_language("Questo è un testo in italiano molto chiaro.")
    assert lang_it == "it"
    
    lang_en = tokenizer.detect_language("This is a very clear English text.")
    assert lang_en == "en"
    print("Language detection verified.")
