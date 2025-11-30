"""OCR processor for extracting text from scanned PDFs using Tesseract."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing for scanned PDFs using Tesseract (fully offline)."""

    def __init__(self, config):
        """Initialize OCR processor with configuration.

        Args:
            config: Configuration object with OCR settings
        """
        self.enabled = config.get("ocr.enabled", True)
        self.tesseract_cmd = config.get("ocr.tesseract_cmd", "tesseract")
        self.lang = config.get("ocr.language", "eng")

        if self.enabled:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                logger.info(f"OCR initialized with Tesseract at: {self.tesseract_cmd}")
            except ImportError:
                logger.warning("pytesseract not installed. OCR will be disabled.")
                self.enabled = False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from a PDF, falling back to OCR when needed.

        This method first attempts normal text extraction using pdfplumber.
        If no text is found (indicating a scanned PDF), it falls back to OCR
        using Tesseract. All processing happens locally without internet access.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text from the PDF, or None if extraction fails
        """
        if not self.enabled:
            logger.debug("OCR is disabled, skipping OCR fallback")
            return None

        try:
            # Try normal extraction first (pdfplumber)
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            if text.strip():
                logger.debug(f"Extracted text from {pdf_path} using pdfplumber (digital PDF)")
                return text

            # Fallback to OCR for scanned PDFs
            logger.info(f"No text found in {pdf_path}, using OCR fallback")
            return self._extract_with_ocr(pdf_path)

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def _extract_with_ocr(self, pdf_path: str) -> Optional[str]:
        """Extract text using OCR (Tesseract).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            OCR-extracted text, or None if OCR fails
        """
        try:
            import pytesseract
            from pdf2image import convert_from_path

            # Convert PDF pages to images
            images = convert_from_path(pdf_path)

            # Run OCR on each page
            ocr_text = []
            for i, img in enumerate(images):
                page_text = pytesseract.image_to_string(img, lang=self.lang)
                if page_text.strip():
                    ocr_text.append(f"--- Page {i+1} ---\n{page_text}")

            result = "\n\n".join(ocr_text)
            logger.info(f"OCR extracted {len(result)} characters from {len(images)} pages")
            return result

        except ImportError as e:
            logger.error(f"Missing OCR dependencies: {e}. Install pytesseract and pdf2image.")
            return None
        except Exception as e:
            logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return None
