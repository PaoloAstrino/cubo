"""Simplemma-based lemmatizer for multilingual support.

Wraps simplemma functionality to provide easy lemmatization for
supported languages (IT, EN, DE, FR, ES, etc.).
"""

from typing import List, Optional, Union
import logging

try:
    import simplemma

    SIMPLEMMA_AVAILABLE = True
except ImportError:
    SIMPLEMMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SimplemmaLemmatizer:
    """Wrapper for simplemma lemmatization."""

    DEFAULT_LANGS = ["it", "en", "de", "fr", "es"]

    def __init__(self, languages: Optional[List[str]] = None):
        """Initialize lemmatizer.

        Args:
            languages: List of language codes to use. If None, uses defaults.
        """
        if not SIMPLEMMA_AVAILABLE:
            logger.warning("simplemma not installed. Lemmatization will be disabled.")

        self.languages = languages or self.DEFAULT_LANGS
        # Validate languages against simplemma supported list if possible,
        # but simplemma is quite permissive.

    def is_available(self) -> bool:
        """Check if simplemma is available."""
        return SIMPLEMMA_AVAILABLE

    def lemmatize_token(self, token: str, lang: Optional[Union[str, List[str]]] = None) -> str:
        """Lemmatize a single token.

        Args:
            token: The word to lemmatize.
            lang: Specific language code or list of codes. If None, uses instance defaults.

        Returns:
            The lemmatized token.
        """
        if not SIMPLEMMA_AVAILABLE:
            return token

        langs = lang or self.languages
        if isinstance(langs, str):
            langs = (langs,)
        else:
            langs = tuple(langs)

        return simplemma.lemmatize(token, lang=langs)

    def lemmatize_text(self, text: str, lang: Optional[Union[str, List[str]]] = None) -> List[str]:
        """Lemmatize a full text string.

        Args:
            text: The text to process.
            lang: Specific language code or list of codes.

        Returns:
            List of lemmatized tokens.
        """
        if not SIMPLEMMA_AVAILABLE:
            return text.split()

        langs = lang or self.languages
        if isinstance(langs, str):
            langs = (langs,)
        else:
            langs = tuple(langs)

        # simplemma.text_lemmatizer returns a list of tokens
        return simplemma.text_lemmatizer(text, lang=langs)
