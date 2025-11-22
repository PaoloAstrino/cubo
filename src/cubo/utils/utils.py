import os
import re
from collections import defaultdict
from functools import wraps
from typing import List, Optional

from src.cubo.utils.logger import logger

# Lazy import for transformers - only import when needed
AutoTokenizer = None


def log_errors(success_msg: str = None, error_prefix: str = "Error"):
    """
    Decorator to handle common error logging pattern.

    Args:
        success_msg: Message to log on success (optional)
        error_prefix: Prefix for error messages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if success_msg:
                    logger.info(success_msg)
                return result
            except Exception as e:
                logger.error(f"{error_prefix} in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


class Utils:
    """Utility functions for CUBO."""

    @staticmethod
    @log_errors("Path sanitized successfully")
    def sanitize_path(path: str, base_dir: str) -> str:
        """Sanitize and validate file path to prevent directory traversal."""
        abs_path = os.path.abspath(path)
        base_abs = os.path.abspath(base_dir)
        if not abs_path.startswith(base_abs):
            raise ValueError("Path traversal detected.")
        return abs_path

    @staticmethod
    @log_errors("File size validated successfully")
    def validate_file_size(file_path: str, max_size_mb: float) -> None:
        """Validate file size against maximum allowed size."""
        size = os.path.getsize(file_path) / (1024 * 1024)
        if size > max_size_mb:
            raise ValueError(f"File size {size:.2f}MB exceeds limit {max_size_mb}MB.")

    @staticmethod
    @log_errors("File type validated successfully")
    def validate_file_type(file_path: str, allowed_extensions: List[str]) -> None:
        """Validate file type based on extension."""
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [e.lower() for e in allowed_extensions]:
            raise ValueError(f"File type {ext} not allowed. Allowed: {allowed_extensions}")

    @staticmethod
    @log_errors("Text cleaned successfully")
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace, normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    @log_errors("Text preprocessed successfully")
    def preprocess_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
        """Advanced text preprocessing: lowercase, remove punctuation, etc."""
        if lowercase:
            text = text.lower()
        if remove_punct:
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = Utils.clean_text(text)  # Reuse clean_text
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks with adaptive sizing based on text length."""
        text_length = len(text)

        # Get adaptive chunk parameters
        chunk_size, overlap = Utils._get_adaptive_chunk_params(text_length, chunk_size, overlap)

        chunks = Utils._create_overlapping_chunks(text, chunk_size, overlap)
        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _get_adaptive_chunk_params(text_length: int, chunk_size: int = None, overlap: int = None) -> tuple:
        """Determine adaptive chunk size and overlap based on text length."""
        if chunk_size is None:
            if text_length < 1000:
                chunk_size = 200
            elif text_length < 5000:
                chunk_size = 500
            else:
                chunk_size = 1000

        if overlap is None:
            if text_length < 1000:
                overlap = 50
            elif text_length < 5000:
                overlap = 100
            else:
                overlap = 200

        return chunk_size, overlap

    @staticmethod
    def _create_overlapping_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from text."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Lightweight sentence splitter using regex."""
        text = re.sub(r'\s+', ' ', text.strip())
        # Split after terminal punctuation followed by space
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _token_count(text: str, tokenizer=None) -> int:
        """Return approximate token count; use HF tokenizer if provided, else fallback to word count."""
        if tokenizer:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: approximate by words
        return max(1, len(text.split()))

    @staticmethod
    @log_errors("Sentence window chunks created successfully")
    def create_sentence_window_chunks(
        text: str,
        window_size: int = 3,
        tokenizer_name: Optional[str] = None
    ) -> List[dict]:
        """
        Create sentence window chunks: single sentences with window metadata.
        Each chunk contains one sentence for matching, plus surrounding context window.

        Args:
            text: Input text to chunk
            window_size: Number of sentences in the context window (odd numbers work best)
            tokenizer_name: Path to HF tokenizer for accurate token counting

        Returns:
            List of dicts with 'text', 'window', and metadata
        """
        try:
            sentences = Utils._split_into_sentences(text)
            if not sentences:
                return []

            # Load tokenizer if provided
            tokenizer = None
            if tokenizer_name:
                try:
                    # Lazy import of transformers
                    if AutoTokenizer is None:
                        from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
                except Exception as e:
                    logger.warning(f"Could not load tokenizer {tokenizer_name}: {e}")

            chunks = []
            n = len(sentences)

            for i, sentence in enumerate(sentences):
                # Calculate window bounds (symmetric around current sentence)
                half_window = window_size // 2
                start = max(0, i - half_window)
                end = min(n, i + half_window + 1)

                # Create window text
                window_sentences = sentences[start:end]
                window_text = " ".join(window_sentences)

                # Calculate token counts
                sentence_tokens = Utils._token_count(sentence, tokenizer)
                window_tokens = Utils._token_count(window_text, tokenizer)

                chunks.append({
                    "text": sentence,  # Single sentence for embedding/matching
                    "window": window_text,  # Full window for context
                    "sentence_index": i,
                    "window_start": start,
                    "window_end": end - 1,
                    "sentence_token_count": sentence_tokens,
                    "window_token_count": window_tokens
                })

            logger.info(f"Created {len(chunks)} sentence window chunks with window_size={window_size}")
            return chunks
        except Exception as e:
            logger.error(f"Error creating sentence window chunks: {e}")
            raise


class Metrics:
    """Basic performance monitoring for enterprise dashboards."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_time(self, operation: str, duration: float):
        """Record operation duration."""
        self.metrics[f"{operation}_time"].append(duration)
        logger.info(f"METRICS: {operation} took {duration:.2f}s")

    def record_count(self, operation: str):
        """Record operation count."""
        count_key = f"{operation}_count"
        if count_key not in self.metrics:
            self.metrics[count_key] = 0
        self.metrics[count_key] += 1
        logger.info(f"METRICS: {operation} count: {self.metrics[count_key]}")

    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        times = self.metrics.get(f"{operation}_time", [])
        return sum(times) / len(times) if times else 0.0

    def get_count(self, operation: str) -> int:
        """Get count for an operation."""
        return self.metrics.get(f"{operation}_count", 0)


# Global metrics instance
metrics = Metrics()
