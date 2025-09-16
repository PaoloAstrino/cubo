import os
import re
from typing import List
from logger import logger

class Utils:
    """Utility functions for CUBO."""

    @staticmethod
    def sanitize_path(path: str, base_dir: str) -> str:
        """Sanitize and validate file path to prevent directory traversal."""
        try:
            abs_path = os.path.abspath(path)
            base_abs = os.path.abspath(base_dir)
            if not abs_path.startswith(base_abs):
                raise ValueError("Path traversal detected.")
            logger.info(f"Path sanitized: {abs_path}")
            return abs_path
        except Exception as e:
            logger.error(f"Error sanitizing path {path}: {e}")
            raise

    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: float) -> None:
        """Validate file size against maximum allowed size."""
        try:
            size = os.path.getsize(file_path) / (1024 * 1024)
            if size > max_size_mb:
                raise ValueError(f"File size {size:.2f}MB exceeds limit {max_size_mb}MB.")
            logger.info(f"File size validated: {file_path} ({size:.2f}MB)")
        except Exception as e:
            logger.error(f"Error validating file size for {file_path}: {e}")
            raise

    @staticmethod
    def validate_file_type(file_path: str, allowed_extensions: List[str]) -> None:
        """Validate file type based on extension."""
        try:
            _, ext = os.path.splitext(file_path)
            if ext.lower() not in [e.lower() for e in allowed_extensions]:
                raise ValueError(f"File type {ext} not allowed. Allowed: {allowed_extensions}")
            logger.info(f"File type validated: {file_path}")
        except Exception as e:
            logger.error(f"Error validating file type for {file_path}: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content."""
        try:
            # Remove extra whitespace, normalize
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            logger.info("Text cleaned successfully")
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise

    @staticmethod
    def preprocess_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
        """Advanced text preprocessing: lowercase, remove punctuation, etc."""
        try:
            if lowercase:
                text = text.lower()
            if remove_punct:
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = Utils.clean_text(text)  # Reuse clean_text
            logger.info("Text preprocessed successfully")
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            raise

    @staticmethod
    def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks with adaptive sizing based on text length."""
        try:
            text_length = len(text)
            
            # Adaptive chunk sizing based on text length
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
            
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap
                if start >= len(text):
                    break
            logger.info(f"Text chunked into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
