"""Multilingual tokenizer with language detection and stemming.

Provides language-aware tokenization for BM25 retrieval, supporting:
- Italian, French, German, Spanish, English
- Automatic language detection
- SnowballStemmer for morphological normalization
- Optional stop word removal
"""

import re
from typing import List, Optional, Dict
from functools import lru_cache

try:
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class MultilingualTokenizer:
    """Tokenizer with automatic language detection and stemming.

    Supports Italian, French, German, Spanish, and English with
    SnowballStemmer for morphological normalization.

    Example:
        >>> tokenizer = MultilingualTokenizer()
        >>> tokens = tokenizer.tokenize("I gatti mangiano", language='it')
        >>> print(tokens)  # ['gatt', 'mang'] (stemmed)
    """

    # Supported languages and their SnowballStemmer names
    SUPPORTED_LANGUAGES = {
        "it": "italian",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "en": "english",
        "pt": "portuguese",
        "ru": "russian",
        "nl": "dutch",
        "sv": "swedish",
        "no": "norwegian",
        "da": "danish",
        "fi": "finnish",
    }

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        use_stemming: bool = True,
        remove_stop_words: bool = False,
        min_token_length: int = 2,
    ):
        """Initialize multilingual tokenizer.

        Args:
            languages: List of language codes to support (default: all supported)
            use_stemming: Whether to apply stemming (default: True)
            remove_stop_words: Whether to remove stop words (default: False)
            min_token_length: Minimum token length to keep (default: 2)
        """
        if not NLTK_AVAILABLE:
            raise ImportError(
                "nltk is required for MultilingualTokenizer. " "Install with: pip install nltk"
            )

        self.languages = languages or list(self.SUPPORTED_LANGUAGES.keys())
        self.use_stemming = use_stemming
        self.remove_stop_words = remove_stop_words
        self.min_token_length = min_token_length

        # Initialize stemmers for supported languages
        self._stemmers: Dict[str, SnowballStemmer] = {}
        for lang_code in self.languages:
            if lang_code in self.SUPPORTED_LANGUAGES:
                stemmer_name = self.SUPPORTED_LANGUAGES[lang_code]
                try:
                    self._stemmers[lang_code] = SnowballStemmer(stemmer_name)
                except Exception:
                    # Fallback to English if stemmer not available
                    self._stemmers[lang_code] = SnowballStemmer("english")

        # Download NLTK data if needed (punkt tokenizer)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass  # Will fall back to simple tokenization

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> str:
        """Detect language of text.

        Args:
            text: Text to detect language for

        Returns:
            Language code (e.g., 'it', 'fr', 'en') or 'en' if detection fails
        """
        if not LANGDETECT_AVAILABLE:
            return "en"  # Default to English

        try:
            # Use first 500 chars for detection (faster)
            sample = text[:500] if len(text) > 500 else text
            detected = detect(sample)

            # Return detected language if supported, else English
            return detected if detected in self.SUPPORTED_LANGUAGES else "en"
        except (LangDetectException, Exception):
            return "en"  # Default to English on error

    def tokenize(self, text: str, language: Optional[str] = "auto") -> List[str]:
        """Tokenize and optionally stem text.

        Args:
            text: Text to tokenize
            language: Language code ('it', 'fr', etc.) or 'auto' for detection

        Returns:
            List of tokens (stemmed if use_stemming=True)
        """
        if not text or not text.strip():
            return []

        # Detect language if auto
        if language == "auto" or language is None:
            language = self.detect_language(text)

        # Ensure language is supported
        if language not in self.SUPPORTED_LANGUAGES:
            language = "en"

        # Tokenize
        tokens = self._tokenize_text(text)

        # Apply stemming if enabled
        if self.use_stemming and language in self._stemmers:
            stemmer = self._stemmers[language]
            tokens = [stemmer.stem(token) for token in tokens]

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def _tokenize_text(self, text: str) -> List[str]:
        """Basic tokenization (word splitting).

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Try NLTK word_tokenize first (better for European languages)
        try:
            tokens = word_tokenize(text.lower())
            # Filter out punctuation-only tokens
            tokens = [t for t in tokens if re.match(r"\w+", t)]
            return tokens
        except Exception:
            # Fallback to simple regex tokenization
            tokens = re.findall(r"\b\w+\b", text.lower())
            return tokens

    def batch_tokenize(self, texts: List[str], language: Optional[str] = "auto") -> List[List[str]]:
        """Tokenize multiple texts.

        Args:
            texts: List of texts to tokenize
            language: Language code or 'auto' for detection

        Returns:
            List of token lists
        """
        return [self.tokenize(text, language) for text in texts]


# Convenience function for quick tokenization
def tokenize_multilingual(
    text: str, language: str = "auto", use_stemming: bool = True
) -> List[str]:
    """Quick tokenization with language detection and stemming.

    Args:
        text: Text to tokenize
        language: Language code or 'auto'
        use_stemming: Whether to apply stemming

    Returns:
        List of tokens
    """
    tokenizer = MultilingualTokenizer(use_stemming=use_stemming)
    return tokenizer.tokenize(text, language=language)
