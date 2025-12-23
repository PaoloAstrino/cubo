"""Advanced PDF parser using PyMuPDF (fitz) and EasyOCR.

Handles complex layouts (tables, columns) and scanned documents via OCR.
"""

import logging
from typing import List

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from easyocr import Reader

    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


logger = logging.getLogger(__name__)


class AdvancedPDFParser:
    """PDF parser using PyMuPDF and EasyOCR."""

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """Initialize parser.

        Args:
            languages: List of languages for OCR (default: ['it', 'en'])
            gpu: Whether to use GPU for OCR (default: False)
        """
        self.languages = languages or ["it", "en"]
        self.gpu = gpu
        self.reader = None

        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF (fitz) not installed. Advanced parsing disabled.")

        if not EASYOCR_AVAILABLE:
            logger.warning("EasyOCR not installed. OCR will be disabled.")

    def _get_reader(self):
        """Lazy load EasyOCR reader."""
        if self.reader is None and EASYOCR_AVAILABLE:
            logger.info(f"Loading EasyOCR reader for {self.languages} (GPU={self.gpu})...")
            self.reader = Reader(self.languages, gpu=self.gpu)
        return self.reader

    def parse(self, pdf_path: str) -> str:
        """Parse PDF file, handling both digital and scanned content.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Extracted text.
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for AdvancedPDFParser")

        doc = fitz.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc):
            logger.debug(f"Processing page {page_num + 1}/{len(doc)}")

            # 1. Try standard text extraction (preserves layout better than pypdf)
            text = page.get_text("text")

            # 2. Check if page is likely scanned (little text, has images)
            if self._is_scanned(page, text):
                logger.info(f"Page {page_num + 1} appears scanned. Attempting OCR...")
                ocr_text = self._perform_ocr(page)
                if ocr_text:
                    text = ocr_text

            full_text.append(text)

        return "\n\n".join(full_text)

    def _is_scanned(self, page, text: str) -> bool:
        """Determine if a page is likely scanned."""
        # Heuristic: If text is very short but page has images covering significant area
        if len(text.strip()) > 50:
            return False

        # Check for images
        images = page.get_images()
        if not images:
            return False

        # Could add more sophisticated coverage check here
        return True

    def _perform_ocr(self, page) -> str:
        """Run EasyOCR on the page image."""
        if not EASYOCR_AVAILABLE:
            return ""

        reader = self._get_reader()
        if not reader:
            return ""

        # Render page to image (pixmap)
        # matrix=fitz.Matrix(2, 2) doubles resolution for better OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        # Convert to bytes
        img_data = pix.tobytes("png")

        # EasyOCR can read from bytes directly
        try:
            # detail=0 returns just the text list
            result = reader.readtext(img_data, detail=0)
            return "\n".join(result)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
