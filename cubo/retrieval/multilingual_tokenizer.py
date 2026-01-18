"""Multilingual tokenizer with language detection and stemming.

Provides language-aware tokenization for BM25 retrieval, supporting:
- Italian, French, German, Spanish, English
- Automatic language detection
- SnowballStemmer for morphological normalization
- Optional stop word removal
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional

try:
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from langdetect import LangDetectException, detect

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
        use_compound_splitter: bool = False,
    ):
        """Initialize multilingual tokenizer.

        Args:
            languages: List of language codes to support (default: all supported)
            use_stemming: Whether to apply stemming (default: True)
            remove_stop_words: Whether to remove stop words (default: False)
            min_token_length: Minimum token length to keep (default: 2)
            use_compound_splitter: Whether to use German compound splitter (default: False)
        """
        if not NLTK_AVAILABLE:
            raise ImportError(
                "nltk is required for MultilingualTokenizer. " "Install with: pip install nltk"
            )

        self.languages = languages or list(self.SUPPORTED_LANGUAGES.keys())
        self.use_stemming = use_stemming
        self.remove_stop_words = remove_stop_words
        self.min_token_length = min_token_length
        self.use_compound_splitter = use_compound_splitter

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

        # Avoid unreliable detection on very short inputs (e.g., single word
        # queries) — default to English for short inputs where language detection
        # is prone to misclassification.
        try:
            token_count = len(re.findall(r"\w+", text))
            # Relaxed constraint: try detection even for single words, but rely on
            # langdetect's internal probability (implied) or accept risk for better retrieval
            if token_count < 1:
                return "en"
        except Exception:
            pass

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

        # Apply German compound splitting if enabled
        if self.use_compound_splitter and language == 'de':
            tokens = self._split_german_compounds(tokens)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens
    
    def _split_german_compounds(self, tokens: List[str]) -> List[str]:
        """Split German compound words.
        
        German frequently uses compound words (e.g., 'Datenschutzgrundverordnung'
        = 'Datenschutz' + 'grund' + 'verordnung'). This method attempts to split
        them for better retrieval.
        
        Args:
            tokens: List of tokens to process
        
        Returns:
            Expanded list with compound words split
        """
        expanded_tokens = []
        
        for token in tokens:
            # Only split long words (compounds are typically long)
            if len(token) < 10:
                expanded_tokens.append(token)
                continue
            
            # Simple heuristic: split on common German compound patterns
            # This is a basic implementation; production could use CharSplit library
            splits = self._heuristic_german_split(token)
            
            if splits and len(splits) > 1:
                # Add both original and split forms
                expanded_tokens.append(token)
                expanded_tokens.extend(splits)
            else:
                expanded_tokens.append(token)
        
        return expanded_tokens
    
    def _heuristic_german_split(self, word: str) -> List[str]:
        """Heuristic German compound word splitting with common-suffix fallbacks.

        Strategy (simple, robust):
        1. Try to match common German suffixes (e.g. 'verordnung', 'schutz').
        2. If not found, fall back to fugenelement-based splitting (original heuristic).
        3. Return lowercase tokens and preserve the original form alongside splits.
        """
        if len(word) < 10:
            return [word]

        w = word.lower()

        # Fast path: common German suffixes / morphemes that often end compounds
        common_suffixes = [
            'verordnung', 'vertrautheit', 'datenschutz', 'schutz', 'ordnung', 'zeit', 'wahl',
            'rechnung', 'gesetz', 'grund', 'steuer', 'arbeit', 'schiff', 'dampf', 'stelle', 'system'
        ]
        for suf in common_suffixes:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                prefix = w[: len(w) - len(suf)]
                # If prefix still long, try to split prefix further on 'grund' etc.
                if prefix.endswith('grund') and len(prefix) - 5 >= 4:
                    return [w, prefix[:-5], 'grund', suf]
                return [w, prefix, suf]

        # Fallback: original fugenelement heuristic (improved bounds)
        linking_elements = ['s', 'n', 'es', 'en', 'e', 'er']
        candidates = []
        for i in range(4, len(w) - 4):  # allow slightly earlier splits
            for link in linking_elements:
                if w[i:i+len(link)] == link:
                    prefix = w[:i]
                    suffix = w[i+len(link):]
                    if len(prefix) >= 4 and len(suffix) >= 4:
                        candidates.append((prefix, suffix))

        if candidates:
            best_split = max(candidates, key=lambda x: len(x[0]))
            # return original + split parts for robustness
            return [w, best_split[0], best_split[1]]

        # As a last attempt, try greedy substring matching for commonly-observed parts
        greedy_parts = []
        for part_len in range(12, 3, -1):
            for i in range(0, len(w) - part_len + 1):
                substr = w[i : i + part_len]
                # heuristic: substrings containing 'schutz'/'ver'/'wahl' indicate morphemes
                if any(x in substr for x in ('schutz', 'ver', 'wahl', 'grund', 'dampf', 'schiff')):
                    left = w[:i]
                    right = w[i+part_len:]
                    if len(left) >= 3 and len(right) >= 3:
                        greedy_parts = [w, left, substr, right]
                        break
            if greedy_parts:
                return greedy_parts

        return [w]

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
