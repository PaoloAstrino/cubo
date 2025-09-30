#!/usr/bin/env python3
"""
Enhanced Document Processor for CUBO
Combines Dolphin vision parsing with EmbeddingGemma-300M semantic embeddings
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from .dolphin_processor import DolphinProcessor
from .model_loader import ModelManager
from .utils import Utils

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor:
    """
    Enhanced document processor combining Dolphin vision parsing
    with EmbeddingGemma-300M semantic embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced document processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Initialize components
        self.dolphin = None
        self.embedding_model = None

        # Try to load Dolphin (optional)
        try:
            self.dolphin = DolphinProcessor()
            logger.info("Dolphin processor loaded")
        except Exception as e:
            logger.warning(f"Dolphin not available: {e}")
            logger.info("Falling back to text-only processing")

        # Load embedding model
        model_loader = ModelManager()
        self.embedding_model = model_loader.load_model()
        logger.info("EmbeddingGemma-300M loaded")

    def process_pdf_with_dolphin(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF using Dolphin vision parsing + EmbeddingGemma embeddings.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of processed chunks with embeddings
        """
        if not self.dolphin:
            raise ValueError("Dolphin processor not available")

        try:
            # Convert PDF to images
            from pdf2image import convert_from_path
            logger.info(f"Converting PDF to images: {pdf_path}")

            images = convert_from_path(pdf_path, dpi=300)
            logger.info(f"Converted to {len(images)} pages")

            # Process each page with Dolphin
            page_contents = self.dolphin.process_document_pages(images)

            # Combine and chunk content
            full_content = "\n\n".join(f"--- Page {i+1} ---\n{content}"
                                       for i, content in enumerate(page_contents))

            # Create chunks with embeddings
            chunks = self._create_enhanced_chunks(full_content, pdf_path)

            logger.info(f"Processed PDF into {len(chunks)} enhanced chunks")
            return chunks

        except ImportError:
            raise ImportError("pdf2image not installed. Run: pip install pdf2image")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def process_image_with_dolphin(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process single image using Dolphin + embeddings.

        Args:
            image_path: Path to image file

        Returns:
            List of processed chunks with embeddings
        """
        if not self.dolphin:
            raise ValueError("Dolphin processor not available")

        try:
            # Load image
            image = Image.open(image_path)

            # Process with Dolphin
            content = self.dolphin.process_image(image)

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
        chunk_size = self.config.get('chunk_size', 512)
        overlap = self.config.get('chunk_overlap', 50)

        text_chunks = Utils.chunk_text(content, chunk_size, overlap)

        # Generate embeddings for each chunk
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Get embedding
            embedding = self.embedding_model.encode(chunk_text, convert_to_numpy=True)

            chunk = {
                'id': f"{Path(source_path).stem}_chunk_{i}",
                'text': chunk_text,
                'embedding': embedding,
                'source': source_path,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'processing_method': 'dolphin_enhanced'
            }
            chunks.append(chunk)

        return chunks

    def process_text_fallback(self, text_path: str) -> List[Dict[str, Any]]:
        """
        Fallback text processing without Dolphin.

        Args:
            text_path: Path to text file

        Returns:
            List of processed chunks with embeddings
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
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
        if file_path.suffix.lower() == '.pdf':
            if self.dolphin:
                return self.process_pdf_with_dolphin(str(file_path))
            else:
                raise ValueError("PDF processing requires Dolphin model")
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            if self.dolphin:
                return self.process_image_with_dolphin(str(file_path))
            else:
                raise ValueError("Image processing requires Dolphin model")
        elif file_path.suffix.lower() == '.txt':
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
        if not self.dolphin:
            return {"error": "Dolphin processor not available"}

        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg']:
            try:
                # Convert to image if PDF
                if file_path.suffix.lower() == '.pdf':
                    from pdf2image import convert_from_path
                    images = convert_from_path(str(file_path), dpi=300, first_page=1, last_page=1)
                    image = images[0] if images else None
                else:
                    image = Image.open(file_path)

                if image:
                    return self.dolphin.extract_structured_data(image)
                else:
                    return {"error": "Could not load image"}

            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Structured extraction requires image/PDF input"}

    def is_dolphin_available(self) -> bool:
        """Check if Dolphin processor is available."""
        return self.dolphin is not None and self.dolphin.is_available()
