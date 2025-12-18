"""
Async Document Processor for CUBO
Provides background document processing with progress tracking.
"""

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class DocumentProcessorWorker(QObject):
    """Worker for async document processing with progress signals."""

    # Signals for GUI communication
    progress_updated = Signal(int, str)  # progress %, status message
    processing_finished = Signal(list)  # results (chunks)
    error_occurred = Signal(str)  # error message
    document_processed = Signal(str, int)  # filename, chunk_count

    def __init__(
        self,
        file_paths: List[str],
        processor_type: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize document processor worker.

        Args:
            file_paths: List of file paths to process
            processor_type: "auto", "enhanced", or "standard"
            config: Configuration dictionary
        """
        super().__init__()
        self.file_paths = file_paths
        self.processor_type = processor_type
        self.config = config or {}
        self.is_cancelled = False

        # Initialize processors lazily
        self.document_loader = None
        self.enhanced_processor = None

    def cancel(self):
        """Cancel processing."""
        self.is_cancelled = True
        logger.info("Document processing cancelled by user")

    def run(self):
        """Main processing function - runs in background thread."""
        try:
            # Setup processing environment
            start_time, all_chunks, total_files = self._setup_processing()

            # Process all documents
            all_chunks = self._process_document_batch(all_chunks, total_files, start_time)

            # Handle successful completion
            if not self.is_cancelled:
                self._handle_processing_completion(all_chunks, total_files, start_time)

        except Exception as e:
            # Handle critical errors
            self._handle_processing_error(e)

    def _setup_processing(self) -> tuple:
        """Setup the processing environment and return initial state."""
        start_time = time.time()
        all_chunks = []
        total_files = len(self.file_paths)

        self.progress_updated.emit(0, "Initializing processors...")
        self._initialize_processors()

        return start_time, all_chunks, total_files

    def _process_document_batch(
        self, all_chunks: List, total_files: int, start_time: float
    ) -> List:
        """Process all documents in the batch."""
        for i, file_path in enumerate(self.file_paths):
            if self.is_cancelled:
                logger.info("Processing cancelled")
                break

            file_start_time = time.time()
            filename = file_path.split("/")[-1].split("\\")[-1]

            self.progress_updated.emit(int((i / total_files) * 100), f"Processing {filename}...")

            try:
                # Process single document
                chunks = self._process_single_document(file_path)

                if chunks:
                    all_chunks.extend(chunks)
                    processing_time = time.time() - file_start_time
                    logger.info(
                        f"Processed {filename}: {len(chunks)} chunks in {processing_time:.2f}s"
                    )

                    # Signal individual document completion
                    self.document_processed.emit(filename, len(chunks))
                else:
                    logger.warning(f"No chunks generated for {filename}")

            except Exception as e:
                error_msg = f"Failed to process {filename}: {str(e)}"
                logger.error(error_msg)
                # Continue with other files instead of failing completely
                self.progress_updated.emit(
                    int((i / total_files) * 100), f"Error processing {filename}, continuing..."
                )

        return all_chunks

    def _handle_processing_completion(self, all_chunks: List, total_files: int, start_time: float):
        """Handle successful completion of processing."""
        total_time = time.time() - start_time
        self.progress_updated.emit(
            100, f"Processing complete! {len(all_chunks)} total chunks in {total_time:.2f}s"
        )
        self.processing_finished.emit(all_chunks)
        logger.info(
            f"Document processing completed: {len(all_chunks)} chunks from {total_files} files"
        )

    def _handle_processing_error(self, error: Exception):
        """Handle critical processing errors."""
        error_msg = f"Critical processing error: {str(error)}"
        logger.error(error_msg)
        self.error_occurred.emit(error_msg)

    def _initialize_processors(self):
        """Initialize document processors."""
        try:
            from cubo.ingestion.document_loader import DocumentLoader

            self.document_loader = DocumentLoader()

            # Initialize enhanced processor if needed
            if self.processor_type in ["auto", "enhanced"]:
                try:
                    from cubo.ingestion.enhanced_document_processor import (
                        EnhancedDocumentProcessor,
                    )

                    self.enhanced_processor = EnhancedDocumentProcessor(self.config)
                    logger.info("Enhanced document processor initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize enhanced processor: {e}")
                    if self.processor_type == "enhanced":
                        raise RuntimeError(
                            "Enhanced processing requested but Dolphin not available"
                        )

        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise

    def _process_single_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document."""
        if self.is_cancelled:
            return []

        # Determine processing method
        use_enhanced = self.processor_type == "enhanced" or (
            self.processor_type == "auto" and self.enhanced_processor is not None
        )

        if use_enhanced and self.enhanced_processor:
            return self._process_with_enhanced(file_path)
        else:
            return self._process_with_standard(file_path)

    def _process_with_enhanced(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document with enhanced (Dolphin) processing."""
        try:
            # Use the enhanced processor
            return self.enhanced_processor.process_document(file_path)
        except Exception as e:
            logger.warning(
                f"Enhanced processing failed for {file_path}, falling back to standard: {e}"
            )
            return self._process_with_standard(file_path)

    def _process_with_standard(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document with standard processing."""
        try:
            from cubo.ingestion.document_loader import DocumentLoader

            loader = DocumentLoader()
            return loader.load_single_document(file_path)
        except Exception as e:
            logger.error(f"Standard processing failed for {file_path}: {e}")
            return []


class BatchDocumentProcessor:
    """Utility class for batch document processing operations."""

    def __init__(self, thread_manager=None):
        self.thread_manager = thread_manager

    def process_documents_batch(
        self, file_paths: List[str], processor_type: str = "auto", batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process documents in batches for better resource utilization.

        Args:
            file_paths: List of file paths to process
            processor_type: Processing type ("auto", "enhanced", "standard")
            batch_size: Number of documents to process concurrently

        Returns:
            List of all document chunks
        """
        if not self.thread_manager:
            from cubo.workers.enhanced_thread_manager import get_enhanced_thread_manager

            self.thread_manager = get_enhanced_thread_manager()

        all_chunks = []

        # Process in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]

            # Submit batch processing tasks
            futures = []
            for file_path in batch:
                if processor_type == "enhanced":
                    future = self.thread_manager.submit_gpu_task(
                        self._process_single_file_enhanced, file_path
                    )
                else:
                    future = self.thread_manager.submit_cpu_task(
                        self._process_single_file_standard, file_path
                    )
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunks = future.result(timeout=300)  # 5 minute timeout
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")

        return all_chunks

    def _process_single_file_enhanced(self, file_path: str) -> List[Dict[str, Any]]:
        """Process single file with enhanced processing."""
        try:
            from cubo.ingestion.enhanced_document_processor import (
                EnhancedDocumentProcessor,
            )

            processor = EnhancedDocumentProcessor()
            return processor.process_document(file_path)
        except Exception as e:
            logger.warning(f"Enhanced processing failed for {file_path}: {e}")
            return self._process_single_file_standard(file_path)

    def _process_single_file_standard(self, file_path: str) -> List[Dict[str, Any]]:
        """Process single file with standard processing."""
        try:
            from cubo.ingestion.document_loader import DocumentLoader

            loader = DocumentLoader()
            return loader.load_single_document(file_path)
        except Exception as e:
            logger.error(f"Standard processing failed for {file_path}: {e}")
            return []
