#!/usr/bin/env python3
"""
Pre-download models for Docker build to avoid runtime timeouts.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_downloader")

def download_nltk():
    logger.info("Downloading NLTK data...")
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        logger.info("NLTK data downloaded.")
    except ImportError:
        logger.warning("NLTK not installed, skipping.")

def download_easyocr():
    logger.info("Downloading EasyOCR models...")
    try:
        import easyocr
        # This triggers download
        easyocr.Reader(['it', 'en'], gpu=False, download_enabled=True)
        logger.info("EasyOCR models downloaded.")
    except ImportError:
        logger.warning("EasyOCR not installed, skipping.")
    except Exception as e:
        logger.warning(f"EasyOCR download failed: {e}")

def download_sentence_transformers():
    logger.info("Checking Sentence Transformers model...")
    # If we rely on local models/ folder, we might skip this.
    # But if we want to be safe:
    try:
        from sentence_transformers import SentenceTransformer
        # If config points to a model name, we could download it.
        # For now, we assume the local model is used or handled separately.
        pass
    except ImportError:
        pass

if __name__ == "__main__":
    logger.info("Starting model pre-download...")
    download_nltk()
    download_easyocr()
    logger.info("Model pre-download complete.")
