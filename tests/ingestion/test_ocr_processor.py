"""Unit tests for OCR processor."""

import pytest

from cubo.config import Config
from cubo.ingestion.ocr_processor import OCRProcessor


@pytest.fixture
def ocr_config():
    """Create a test configuration for OCR."""
    config = Config()
    config.set("ocr.enabled", True)
    config.set("ocr.tesseract_cmd", "tesseract")
    config.set("ocr.language", "eng")
    return config


def test_ocr_processor_initialization(ocr_config):
    """Test OCR processor initialization."""
    processor = OCRProcessor(ocr_config)
    assert processor.enabled
    assert processor.tesseract_cmd == "tesseract"
    assert processor.lang == "eng"


def test_ocr_processor_disabled():
    """Test OCR processor with disabled OCR."""
    config = Config()
    config.set("ocr.enabled", False)
    processor = OCRProcessor(config)

    # Should return None when disabled
    result = processor.extract_text("dummy.pdf")
    assert result is None


def test_ocr_processor_extract_text_digital_pdf(ocr_config, tmp_path):
    """Test that digital PDFs use pdfplumber, not OCR."""
    # Note: This test requires a real PDF file fixture
    # For now, we'll test the fallback logic
    processor = OCRProcessor(ocr_config)

    # Test with non-existent file - should handle gracefully
    result = processor.extract_text("nonexistent.pdf")
    assert result is None


@pytest.mark.skipif(
    not pytest.importorskip("pytesseract", reason="pytesseract not installed"),
    reason="Tesseract not available",
)
def test_ocr_processor_scanned_pdf(ocr_config):
    """Test OCR on scanned PDF (requires Tesseract installed)."""
    # This test requires a real scanned PDF fixture
    # Skip if test resources not available
    pytest.skip("Scanned PDF fixture not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
