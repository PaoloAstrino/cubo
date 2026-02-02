#!/usr/bin/env python3
"""
Enhanced Document Processor for CUBO
Combines Dolphin vision parsing with EmbeddingGemma-300M semantic embeddings
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.embeddings.model_loader import ModelManager

try:
    from cubo.models.vision_processor import VisionProcessor
except Exception:
    VisionProcessor = None
    # Vision model support is optional; if import fails, fallback occurs at runtime.
from cubo.utils.utils import Utils

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """
    Enhanced document processor combining optional vision parsing
    with EmbeddingGemma-300M semantic embeddings.
    """

    def __init__(self, config: Dict[str, Any], skip_model: bool = False):
        """
        Initialize enhanced document processor.

        Args:
                config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.vision_processor = None
        self.embedding_model = None

        # Try to load vision processor (optional)
        try:
            self.vision_processor = VisionProcessor()
            logger.info("Vision processor loaded")
        except Exception as e:
            logger.warning(f"Vision processor not available: {e}")
            logger.info("Falling back to text-only processing")

        # Load embedding model if not skipping models
        self.embedding_model = None
        if not skip_model:
            model_loader = ModelManager()
            self.embedding_model = model_loader.load_model()
            logger.info("EmbeddingGemma-300M loaded")

    def process_pdf_with_vision(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF using an optional vision model + EmbeddingGemma embeddings.

        Args:
                pdf_path: Path to PDF file

        Returns:
                List of processed chunks with embeddings
        """
        if not self.vision_processor:
            raise ValueError("Vision processor not available")

        try:
            images = self._convert_pdf_to_images(pdf_path)
            page_contents = self.vision_processor.process_document_pages(images)
            full_content = self._combine_page_contents(page_contents)
            chunks = self._create_enhanced_chunks(full_content, pdf_path)

            logger.info(f"Processed PDF into {len(chunks)} enhanced chunks")
            return chunks

        except ImportError:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images."""
        from pdf2image import convert_from_path

        logger.info(f"Converting PDF to images: {pdf_path}")

        images = convert_from_path(pdf_path, dpi=300)
        logger.info(f"Converted to {len(images)} pages")
        return images

    def _combine_page_contents(self, page_contents: List[str]) -> str:
        """Combine page contents into a single document."""
        return "\n\n".join(
            f"--- Page {i+1} ---\n{content}" for i, content in enumerate(page_contents)
        )

    def process_image_with_vision(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process single image using an optional vision model + embeddings.

        Args:
                image_path: Path to image file

        Returns:
                List of processed chunks with embeddings
        """
        if not self.vision_processor:
            raise ValueError("Vision processor not available")

        try:
            # Load image
            image = Image.open(image_path)

            # Process with vision processor
            content = self.vision_processor.process_image(image)

            # Create chunks with embeddings
            chunks = self._create_enhanced_chunks(content, image_path)

            logger.info(f"Processed image into {len(chunks)} enhanced chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def _create_enhanced_chunks(self, content: str, source_path: str) -> List[Dict[str, Any]]:
        """
        Create chunks with semantic embeddings.

        Args:
                content: Extracted text content
                source_path: Source file path

        Returns:
                List of chunk dictionaries with embeddings
        """
        # Chunk the content
        chunk_size = self.config.get("chunk_size", 512)
        overlap = self.config.get("chunk_overlap", 50)

        text_chunks = Utils.chunk_text(content, chunk_size, overlap)

        # Generate embeddings for each chunk
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Get embedding; apply document prompt if model defines it
            try:
                dprefix = EmbeddingGenerator.get_prompt_prefix_for_model(
                    config.get("model_path"), "document"
                )
                text_to_encode = dprefix + chunk_text if dprefix else chunk_text
            except Exception:
                text_to_encode = chunk_text

            embedding = self.embedding_model.encode(text_to_encode, convert_to_numpy=True)

            chunk = {
                "id": f"{Path(source_path).stem}_chunk_{i}",
                "text": chunk_text,
                "embedding": embedding,
                "source": source_path,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "processing_method": "vision_enhanced",
            }
            chunks.append(chunk)

        return chunks

    def process_text_fallback(self, text_path: str) -> List[Dict[str, Any]]:
        """
        Fallback text processing without a vision model.

        Args:
                text_path: Path to text file

        Returns:
                List of processed chunks with embeddings
        """
        try:
            with open(text_path, encoding="utf-8") as f:
                content = f.read()

            chunks = self._create_enhanced_chunks(content, text_path)

            logger.info(f"Processed text file into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process any supported document type.

        Args:
                file_path: Path to document file

        Returns:
                List of processed chunks with embeddings
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine processing method based on file type
        if file_path.suffix.lower() == ".pdf":
            if self.vision_processor:
                return self.process_pdf_with_vision(str(file_path))
            else:
                raise ValueError("PDF processing requires a vision model")
        elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            if self.vision_processor:
                return self.process_image_with_vision(str(file_path))
            else:
                raise ValueError("Image processing requires a vision model")
        elif file_path.suffix.lower() == ".txt":
            return self.process_text_fallback(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def extract_structured_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structured information from document.

        Args:
                file_path: Path to document

        Returns:
                Structured information dictionary
        """
        if not self.vision_processor:
            return {"error": "Vision processor not available"}

        file_path = Path(file_path)

        if not self._is_supported_for_structured_extraction(file_path):
            return {"error": "Structured extraction requires image/PDF input"}

        try:
            image = self._load_image_for_extraction(file_path)
            return (
                self.vision_processor.extract_structured_data(image)
                if image
                else {"error": "Could not load image"}
            )
        except Exception as e:
            return {"error": str(e)}

    def _is_supported_for_structured_extraction(self, file_path: Path) -> bool:
        """Check if file type supports structured extraction."""
        return file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]

    def _load_image_for_extraction(self, file_path: Path) -> Image.Image:
        """Load image for structured extraction."""
        if file_path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            images = convert_from_path(str(file_path), dpi=300, first_page=1, last_page=1)
            return images[0] if images else None
        else:
            return Image.open(file_path)

    def is_vision_available(self) -> bool:
        """Check if a vision processor is available."""
        return self.vision_processor is not None and self.vision_processor.is_available()
