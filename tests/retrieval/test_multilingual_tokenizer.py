"""Unit tests for multilingual tokenizer."""
import pytest
from cubo.retrieval.multilingual_tokenizer import (
    MultilingualTokenizer,
    tokenize_multilingual,
    NLTK_AVAILABLE,
    LANGDETECT_AVAILABLE
)


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
class TestMultilingualTokenizer:
    """Test multilingual tokenizer functionality."""
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = MultilingualTokenizer()
        assert tokenizer.use_stemming is True
        assert len(tokenizer._stemmers) > 0
    
    def test_italian_stemming(self):
        """Test Italian stemming (gatto/gatti -> gatt)."""
        tokenizer = MultilingualTokenizer()
        
        # Test singular and plural
        tokens_singular = tokenizer.tokenize("il gatto mangia", language='it')
        tokens_plural = tokenizer.tokenize("i gatti mangiano", language='it')
        
        # Both should have same stem for cat
        assert 'gatt' in tokens_singular
        assert 'gatt' in tokens_plural
        
        # Both should have same stem for eat
        assert any('mang' in t for t in tokens_singular)
        assert any('mang' in t for t in tokens_plural)
    
    def test_french_stemming(self):
        """Test French stemming (développement/développer)."""
        tokenizer = MultilingualTokenizer()
        
        tokens1 = tokenizer.tokenize("le développement", language='fr')
        tokens2 = tokenizer.tokenize("développer", language='fr')
        
        # Should have similar stems
        assert len(tokens1) > 0
        assert len(tokens2) > 0
    
    def test_german_stemming(self):
        """Test German stemming."""
        tokenizer = MultilingualTokenizer()
        
        tokens = tokenizer.tokenize("die Entwicklung entwickeln", language='de')
        assert len(tokens) > 0
    
    def test_spanish_stemming(self):
        """Test Spanish stemming."""
        tokenizer = MultilingualTokenizer()
        
        tokens = tokenizer.tokenize("los derechos del ciudadano", language='es')
        assert len(tokens) > 0
        assert 'derech' in tokens  # Stemmed form of "derechos"
    
    def test_english_stemming(self):
        """Test English stemming."""
        tokenizer = MultilingualTokenizer()
        
        tokens = tokenizer.tokenize("running runs runner", language='en')
        # All should stem to 'run'
        assert all('run' in t for t in tokens)
    
    @pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
    def test_language_detection_italian(self):
        """Test automatic language detection for Italian."""
        tokenizer = MultilingualTokenizer()
        
        detected = tokenizer.detect_language(
            "Questo è un documento italiano sui diritti dei cittadini"
        )
        assert detected == 'it'
    
    @pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
    def test_language_detection_french(self):
        """Test automatic language detection for French."""
        tokenizer = MultilingualTokenizer()
        
        detected = tokenizer.detect_language(
            "Ceci est un document français sur le développement logiciel"
        )
        assert detected == 'fr'
    
    @pytest.mark.skipif(not LANGDETECT_AVAILABLE, reason="langdetect not installed")
    def test_auto_language_tokenization(self):
        """Test tokenization with automatic language detection."""
        tokenizer = MultilingualTokenizer()
        
        # Italian text
        tokens_it = tokenizer.tokenize(
            "I gatti mangiano il pesce",
            language='auto'
        )
        assert len(tokens_it) > 0
        
        # French text
        tokens_fr = tokenizer.tokenize(
            "Le développement du logiciel",
            language='auto'
        )
        assert len(tokens_fr) > 0
    
    def test_no_stemming(self):
        """Test tokenization without stemming."""
        tokenizer = MultilingualTokenizer(use_stemming=False)
        
        tokens = tokenizer.tokenize("running runs runner", language='en')
        # Should not be stemmed
        assert 'running' in tokens
        assert 'runs' in tokens
        assert 'runner' in tokens
    
    def test_min_token_length(self):
        """Test minimum token length filtering."""
        tokenizer = MultilingualTokenizer(min_token_length=3)
        
        tokens = tokenizer.tokenize("a ab abc abcd", language='en')
        # Only tokens >= 3 chars should remain
        assert 'a' not in tokens
        assert 'ab' not in tokens
        assert len([t for t in tokens if len(t) >= 3]) > 0
    
    def test_empty_text(self):
        """Test handling of empty text."""
        tokenizer = MultilingualTokenizer()
        
        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize("   ") == []
        assert tokenizer.tokenize(None) == []
    
    def test_batch_tokenization(self):
        """Test batch tokenization."""
        tokenizer = MultilingualTokenizer()
        
        texts = [
            "il gatto",
            "le chat",
            "die Katze"
        ]
        
        results = tokenizer.batch_tokenize(texts, language='auto')
        assert len(results) == 3
        assert all(len(tokens) > 0 for tokens in results)
    
    def test_unsupported_language_fallback(self):
        """Test fallback to English for unsupported languages."""
        tokenizer = MultilingualTokenizer()
        
        # Use unsupported language code
        tokens = tokenizer.tokenize("some text", language='xx')
        # Should fall back to English
        assert len(tokens) > 0
    
    def test_convenience_function(self):
        """Test convenience tokenize_multilingual function."""
        tokens = tokenize_multilingual("I gatti mangiano", language='it')
        assert len(tokens) > 0
        assert 'gatt' in tokens


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_multilingual_consistency():
    """Test that same words in different languages get different stems."""
    tokenizer = MultilingualTokenizer()
    
    # "cat" in different languages
    tokens_it = tokenizer.tokenize("gatto", language='it')
    tokens_fr = tokenizer.tokenize("chat", language='fr')
    tokens_de = tokenizer.tokenize("Katze", language='de')
    
    # All should produce tokens
    assert len(tokens_it) > 0
    assert len(tokens_fr) > 0
    assert len(tokens_de) > 0
    
    # Stems should be different
    assert tokens_it[0] != tokens_fr[0]
    assert tokens_fr[0] != tokens_de[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
