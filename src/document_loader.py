import hashlib
import os
from typing import List, Optional
from docx import Document
from PyPDF2 import PdfReader
from src.config import config
from src.utils import Utils
from src.logger import logger


class DocumentLoader:
    """Handles loading and processing of various document types for CUBO."""

    def __init__(self, skip_model: bool = False):
        self.supported_extensions = config.get("supported_extensions", [".txt", ".docx", ".pdf", ".md"])
        self.enhanced_processor = None
        self.skip_model = skip_model

        # Try to load enhanced processor if Dolphin is enabled
        if not self.skip_model and config.get("dolphin", {}).get("enabled", False):
            try:
                from .enhanced_document_processor import EnhancedDocumentProcessor
                self.enhanced_processor = EnhancedDocumentProcessor(config, skip_model=self.skip_model)
                logger.info("Enhanced document processor (Dolphin) loaded")
            except Exception as e:
                logger.warning(f"Enhanced processor not available: {e}")

    def load_single_document(self, file_path: str, chunking_config: dict = None) -> List[dict]:
        """Load and process a single document file with automatic enhanced processing when available."""

        # Try enhanced processing first if available
        if self._should_use_enhanced_processing():
            result = self._try_enhanced_processing(file_path)
            if result is not None:
                return result

        # Fall back to standard processing
        return self._load_and_process_standard(file_path, chunking_config)

    def _should_use_enhanced_processing(self) -> bool:
        """Check if enhanced processing should be used."""
        return (self.enhanced_processor and
                config.get("dolphin", {}).get("enabled", False))

    def _try_enhanced_processing(self, file_path: str) -> Optional[List[dict]]:
        """Try enhanced processing, return None if it fails."""
        try:
            logger.info(f"Using enhanced processing for {file_path}")
            return self.enhanced_processor.process_document(file_path)
        except Exception as e:
            logger.warning(f"Enhanced processing failed, falling back to standard: {e}")
            return None

    def _load_and_process_standard(self, file_path: str, chunking_config: dict = None) -> List[dict]:
        """Load and process document using standard processing."""
        try:
            # Validate file size
            Utils.validate_file_size(file_path, config.get("max_file_size_mb", 10))

            # Load text content
            text = self._load_text_from_file(file_path)
            if not text:
                logger.warning(f"No text content found in {file_path}")
                return []

            # Process and chunk text
            return self._process_and_chunk_text(text, file_path, chunking_config)

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return []

    def _load_text_from_file(self, file_path: str) -> str:
        """Load text content from different file types."""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_ext == '.docx':
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif file_ext == '.pdf':
            reader = PdfReader(file_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_and_chunk_text(self, text: str, file_path: str, chunking_config: dict = None) -> List[dict]:
        """
        Process text content and create chunks using sentence window chunking.

        Args:
            text: The raw text content to process
            file_path: Path to the source file (used for logging)
            chunking_config: Optional configuration for chunking parameters

        Returns:
            List of chunk dictionaries with processed text content
        """
        text = Utils.clean_text(text)
        cfg = self._configure_chunking(chunking_config)
        chunks = self._create_sentence_window_chunks(text, cfg)

        self._embed_file_metadata(chunks, file_path)
        self._log_chunking_results(file_path, chunks, cfg)
        return chunks

    def _configure_chunking(self, chunking_config: dict = None) -> dict:
        """Configure sentence window chunking parameters."""
        cfg = {
            "method": "sentence_window",
            "window_size": 3,
            "tokenizer_name": None
        }

        # Allow overriding window_size if provided
        if chunking_config and "window_size" in chunking_config:
            cfg["window_size"] = chunking_config["window_size"]

        return cfg

    def _create_sentence_window_chunks(self, text: str, cfg: dict) -> List[dict]:
        """Create sentence window chunks using the configured parameters."""
        return Utils.create_sentence_window_chunks(
            text,
            window_size=cfg["window_size"],
            tokenizer_name=cfg["tokenizer_name"]
        )

    def _embed_file_metadata(self, chunks: List[dict], file_path: str) -> None:
        """Add filename, chunk index, and hash metadata to each chunk."""
        if not chunks:
            return

        file_hash = self._compute_file_hash(file_path)
        filename = os.path.basename(file_path)

        for idx, chunk in enumerate(chunks):
            chunk.setdefault("chunk_index", idx)
            chunk["filename"] = filename
            chunk["file_path"] = file_path
            chunk["file_hash"] = file_hash
            chunk["token_count"] = chunk.get("sentence_token_count") or len(chunk.get("text", "").split())

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute a stable MD5 hash for the file contents."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.warning(f"Unable to hash {file_path}: {e}")
            return ""
        return hash_md5.hexdigest()

    def _log_chunking_results(self, file_path: str, chunks: List[dict], cfg: dict):
        """Log the results of the chunking process."""
        logger.info(f"Loaded and chunked {os.path.basename(file_path)} into "
                    f"{len(chunks)} chunks using {cfg['method']} method.")

    def load_documents_from_folder(self, folder_path: str) -> List[dict]:
        """Load all supported documents from a folder, including subfolders."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

        # Find all supported files
        supported_files = self._find_supported_files(folder_path)

        if not supported_files:
            logger.warning(f"No supported files {self.supported_extensions} found in the specified folder or its subfolders.")
            return []

        # Process all files
        return self._process_files_batch(supported_files)

    def _find_supported_files(self, folder_path: str) -> List[str]:
        """
        Find all supported files in the folder and subfolders.

        Args:
            folder_path: Root folder path to search

        Returns:
            List of absolute paths to supported files
        """
        supported_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in self.supported_extensions:
                    supported_files.append(os.path.join(root, file))
        return supported_files

    def _process_files_batch(self, file_paths: List[str]) -> List[dict]:
        """
        Process a batch of files and return combined chunks.

        Args:
            file_paths: List of file paths to process

        Returns:
            Combined list of all chunks from all files
        """
        documents = []
        processing_method = self._determine_processing_method()

        self._log_batch_processing_start(file_paths, processing_method)
        documents = self._process_all_files(file_paths)
        self._log_batch_processing_complete(documents)

        return documents

    def _determine_processing_method(self) -> str:
        """Determine which processing method will be used."""
        return "enhanced" if self._should_use_enhanced_processing() else "standard"

    def _log_batch_processing_start(self, file_paths: List[str], processing_method: str):
        """Log the start of batch processing."""
        logger.info(f"Loading {len(file_paths)} documents using {processing_method} processing...")

    def _process_all_files(self, file_paths: List[str]) -> List[dict]:
        """Process all files and collect their chunks."""
        documents = []
        for file_path in file_paths:
            chunks = self.load_single_document(file_path)
            documents.extend(chunks)
        return documents

    def _log_batch_processing_complete(self, documents: List[dict]):
        """Log the completion of batch processing."""
        logger.info(f"Total documents loaded and chunked into {len(documents)} chunks.")

    def load_documents(self, file_paths: List[str]) -> List[str]:
        """Load multiple documents from a list of file paths."""
        return self._process_files_batch(file_paths)
