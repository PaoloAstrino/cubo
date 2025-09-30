import os
from typing import List
from docx import Document
from PyPDF2 import PdfReader
from src.config import config
from src.utils import Utils
from src.logger import logger


class DocumentLoader:
    """Handles loading and processing of various document types for CUBO."""

    def __init__(self):
        self.supported_extensions = config.get("supported_extensions", [".txt", ".docx", ".pdf", ".md"])
        self.enhanced_processor = None

        # Try to load enhanced processor if Dolphin is enabled
        if config.get("dolphin", {}).get("enabled", False):
            try:
                from .enhanced_document_processor import EnhancedDocumentProcessor
                self.enhanced_processor = EnhancedDocumentProcessor(config)
                logger.info("Enhanced document processor (Dolphin) loaded")
            except Exception as e:
                logger.warning(f"Enhanced processor not available: {e}")

    def load_single_document(self, file_path: str, chunking_config: dict = None) -> List[dict]:
        """Load and process a single document file with automatic enhanced processing when available."""

        # Automatically use enhanced processing if available and enabled in config
        if self.enhanced_processor and config.get("dolphin", {}).get("enabled", False):
            try:
                logger.info(f"Using enhanced processing for {file_path}")
                return self.enhanced_processor.process_document(file_path)
            except Exception as e:
                logger.warning(f"Enhanced processing failed, falling back to standard: {e}")
                # Fall through to standard processing
                logger.warning(f"Enhanced processing failed, falling back to standard: {e}")
                # Fall through to standard processing

        # Standard processing
        Utils.validate_file_size(file_path, config.get("max_file_size_mb", 10))

        text = ""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_ext == '.docx':
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_ext == '.pdf':
                reader = PdfReader(file_path)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if text:
                text = Utils.clean_text(text)

                # Always use sentence window chunking for optimal quality
                cfg = {
                    "method": "sentence_window",
                    "window_size": 3,
                    "tokenizer_name": None
                }
                # Allow overriding window_size if provided
                if chunking_config and "window_size" in chunking_config:
                    cfg["window_size"] = chunking_config["window_size"]

                # Use sentence window chunking
                chunks = Utils.create_sentence_window_chunks(
                    text,
                    window_size=cfg["window_size"],
                    tokenizer_name=cfg["tokenizer_name"]
                )

                logger.info(f"Loaded and chunked {os.path.basename(file_path)} into "
                            f"{len(chunks)} chunks using {cfg['method']} method.")
                return chunks
            else:
                logger.warning(f"No text content found in {file_path}")
                return []

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []
        """Load and process a single document file with configurable chunking."""
        Utils.validate_file_size(file_path, config.get("max_file_size_mb", 10))

        text = ""
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_ext == '.docx':
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_ext == '.pdf':
                reader = PdfReader(file_path)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if text:
                text = Utils.clean_text(text)

                # Always use sentence window chunking for optimal quality
                cfg = {
                    "method": "sentence_window",
                    "window_size": 3,
                    "tokenizer_name": None
                }
                # Allow overriding window_size if provided
                if chunking_config and "window_size" in chunking_config:
                    cfg["window_size"] = chunking_config["window_size"]

                # Use sentence window chunking
                chunks = Utils.create_sentence_window_chunks(
                    text,
                    window_size=cfg["window_size"],
                    tokenizer_name=cfg["tokenizer_name"]
                )

                logger.info(f"Loaded and chunked {os.path.basename(file_path)} into "
                            f"{len(chunks)} chunks using {cfg['method']} method.")
                return chunks
            else:
                logger.warning(f"No text content found in {file_path}")
                return []

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def load_documents_from_folder(self, folder_path: str) -> List[dict]:
        """Load all supported documents from a folder, including subfolders."""
        documents = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

        # Recursively find all supported files
        supported_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in self.supported_extensions:
                    supported_files.append(os.path.join(root, file))

        if not supported_files:
            logger.warning(f"No supported files {self.supported_extensions} found in the specified folder or its subfolders.")
            return []

        processing_method = "enhanced" if (self.enhanced_processor and
                                           config.get("dolphin", {}).get("enabled", False)) else "standard"
        logger.info(f"Loading {len(supported_files)} documents from {folder_path} "
                    f"using {processing_method} processing...")

        for file_path in supported_files:
            chunks = self.load_single_document(file_path)
            documents.extend(chunks)

        logger.info(f"Total documents loaded and chunked into {len(documents)} chunks.")
        return documents

    def load_documents(self, file_paths: List[str]) -> List[str]:
        """Load multiple documents from a list of file paths."""
        documents = []

        processing_method = "enhanced" if (self.enhanced_processor and
                                           config.get("dolphin", {}).get("enabled", False)) else "standard"
        logger.info(f"Loading {len(file_paths)} documents "
                    f"using {processing_method} processing...")

        for file_path in file_paths:
            chunks = self.load_single_document(file_path)
            documents.extend(chunks)

        logger.info(f"Total documents loaded and chunked into {len(documents)} chunks.")
        return documents
