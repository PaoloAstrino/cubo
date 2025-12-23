import os
from unittest.mock import MagicMock, patch

import pytest

from cubo.config import config
from cubo.ingestion.document_loader import DocumentLoader
from cubo.ingestion.pdf_parser import AdvancedPDFParser


# Mock fitz and easyocr since we might not have them installed or want to run them
@pytest.fixture
def mock_fitz():
    with patch("cubo.ingestion.pdf_parser.fitz") as mock:
        yield mock


@pytest.fixture
def mock_easyocr():
    with patch("cubo.ingestion.pdf_parser.Reader") as mock:
        yield mock


def test_advanced_parser_digital_pdf(mock_fitz):
    """Test parsing a digital PDF (text extraction)."""
    # Setup mock
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_fitz.open.return_value = mock_doc
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__len__.return_value = 1

    # Simulate digital text
    mock_page.get_text.return_value = "Invoice #12345\nTotal: 500 EUR"

    parser = AdvancedPDFParser()
    text = parser.parse("dummy.pdf")

    assert "Invoice #12345" in text
    assert "Total: 500 EUR" in text
    mock_page.get_text.assert_called_with("text")


def test_advanced_parser_scanned_pdf(mock_fitz, mock_easyocr):
    """Test parsing a scanned PDF (OCR fallback)."""
    # Setup mock
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_fitz.open.return_value = mock_doc
    mock_doc.__iter__.return_value = [mock_page]
    mock_doc.__len__.return_value = 1

    # Simulate empty text (scanned)
    mock_page.get_text.return_value = "   "
    mock_page.get_images.return_value = [("img_ref",)]  # Has images

    # Mock OCR reader
    mock_reader_instance = MagicMock()
    mock_easyocr.return_value = mock_reader_instance
    mock_reader_instance.readtext.return_value = ["Scanned Content", "Detected"]

    parser = AdvancedPDFParser()
    # Force availability flags for test
    with patch("cubo.ingestion.pdf_parser.PYMUPDF_AVAILABLE", True), patch(
        "cubo.ingestion.pdf_parser.EASYOCR_AVAILABLE", True
    ):

        text = parser.parse("dummy.pdf")

        assert "Scanned Content" in text
        assert "Detected" in text
        # Verify OCR was called
        mock_reader_instance.readtext.assert_called()


def test_document_loader_integration():
    """Test that DocumentLoader uses AdvancedPDFParser when configured."""
    # Mock config.get to return "advanced" for "parser" key
    original_get = config.get

    def mock_get(key, default=None):
        if key == "parser":
            return "advanced"
        return original_get(key, default)

    with patch.object(config, "get", side_effect=mock_get):
        with patch("cubo.ingestion.document_loader.AdvancedPDFParser") as MockParser:
            mock_instance = MockParser.return_value
            mock_instance.parse.return_value = "Parsed Content"

            loader = DocumentLoader()
            # Ensure parser was initialized
            assert loader.parser_type == "advanced"
            assert loader.advanced_parser is not None

            # Test loading
            # We need to mock _load_text_from_file or the specific branch
            # But _load_text_from_file is called internally.
            # Let's mock the file extension check or just call _load_text_from_file directly

            result = loader._load_text_from_file("test.pdf")
            assert result == "Parsed Content"
            mock_instance.parse.assert_called_with("test.pdf")


def test_eu_invoice_sample():
    """Test with the actual sample file if it exists."""
    sample_path = "data/test_invoice.pdf"
    if not os.path.exists(sample_path):
        pytest.skip(f"Sample file {sample_path} not found")

    # This test requires actual dependencies installed
    import importlib

    if importlib.util.find_spec("easyocr") is None or importlib.util.find_spec("fitz") is None:
        pytest.skip("Dependencies not installed")

    parser = AdvancedPDFParser()
    text = parser.parse(sample_path)

    assert len(text) > 0
    # Add specific assertions based on invoice content if known
