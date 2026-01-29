#!/usr/bin/env python3
"""
CuboCore - Pure business logic for CUBO RAG system.

This module contains the core CUBO logic without any CLI dependencies.
It can be safely imported in any context (API, notebooks, tests) without
triggering interactive prompts or side effects.

For CLI usage, see cubo.main.CuboCLI which wraps this class.
"""

import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from cubo.config import config
from cubo.config.settings import settings
from cubo.utils.logger import logger

# Delay importing heavy components until needed to avoid importing optional
# dependencies at module import time (improves CLI responsiveness).


class CuboCore:
    """
    Core CUBO logic - no CLI dependencies.

    This class provides the pure business logic for:
    - Component initialization (model, retriever, generator)
    - Document ingestion and indexing
    - Query retrieval
    - Response generation

    Thread-safe: Uses internal locking for state mutations.

    Example:
        >>> core = CuboCore()
        >>> core.initialize_components()
        >>> core.build_index("./data")
        >>> results = core.query_retrieve("What is RAG?")
    """

    def __init__(self):
        """Initialize CuboCore with empty components."""
        self.model = None
        self.doc_loader = None
        self.retriever = None
        self.generator = None
        self.vector_store = None
        self._state_lock = threading.RLock()

    @property
    def is_initialized(self) -> bool:
        """Check if core components are initialized."""
        return self.model is not None and self.retriever is not None

    def initialize_components(self) -> bool:
        """
        Initialize model and all components.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        logger.info("Loading embedding model... (this may take a few minutes)")
        start_time = time.time()

        try:
            with self._state_lock:
                # Import model_manager lazily to avoid heavy imports during
                # module import time.
                from cubo.embeddings.model_loader import model_manager

                self.model = model_manager.get_model()
            logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

        # Initialize components - protect state mutation
        with self._state_lock:
            # Lazy import of components that may import optional heavy deps
            from cubo.ingestion.document_loader import DocumentLoader
            from cubo.processing.generator import create_response_generator
            from cubo.retrieval.retriever import DocumentRetriever

            self.doc_loader = DocumentLoader()
            # Inject settings directly from Pydantic models
            self.retriever = DocumentRetriever(
                self.model,
                top_k=settings.retrieval.default_top_k,
                window_size=settings.retrieval.default_window_size,
                # TODO: Migrate these flags to RetrievalSettings in a future refactor
                use_sentence_window=config.get("chunking.use_sentence_window", True),
                use_reranker=config.get("retrieval.use_reranker", True),
            )
            self.generator = create_response_generator()
            # Expose vector store from retriever for collection management
            if hasattr(self.retriever, "collection"):
                self.vector_store = self.retriever.collection

        return True

    def build_index(self, data_folder: str = None) -> int:
        """
        Initialize components if needed, load pre-chunked documents and add them to vector DB.

        Args:
            data_folder: Path to folder containing documents.
                         Defaults to config 'data_folder'.

        Returns:
            Number of document chunks processed/added.

        Raises:
            RuntimeError: If component initialization fails.

        Thread-safe: Uses _state_lock to prevent race conditions.
        """
        with self._state_lock:
            # Ensure components are set (model, retriever, generator)
            if not self.model or not self.retriever or not self.generator:
                if not self.initialize_components():
                    raise RuntimeError(
                        "Failed to initialize model and components for index building"
                    )

            folder = data_folder or config.get("data_folder")
            
            # Try to load pre-chunked documents from parquet first
            from pathlib import Path
            deep_output_dir = Path(config.get("ingestion.deep.output_dir", "./storage/deep"))
            chunks_parquet = deep_output_dir / "chunks_deep.parquet"
            
            documents = None
            if chunks_parquet.exists():
                logger.info(f"Loading pre-chunked documents from {chunks_parquet}")
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(chunks_parquet)
                    # Convert parquet table to list of dicts
                    documents = table.to_pylist()
                    logger.info(f"Loaded {len(documents)} pre-chunked documents from parquet")
                except Exception as e:
                    logger.warning(f"Failed to load pre-chunked documents: {e}. Falling back to re-chunking.")
                    documents = None
            
            # Fallback: re-chunk documents if parquet not available
            if documents is None:
                documents = self._load_all_documents(folder)
            
            if not documents:
                return 0

            # Add documents to the vector DB
            self._add_documents_to_db(documents)
            return len(documents)

    def ingest_documents(self, data_folder: str = None) -> int:
        """
        Load and chunk all documents from a folder.

        Note: This does NOT add them to the vector DB.
        Call build_index() to persist to store.

        Args:
            data_folder: Path to folder containing documents.
                         Defaults to config 'data_folder'.

        Returns:
            Number of chunks loaded.
        """
        folder = data_folder or config.get("data_folder")
        if not self.doc_loader:
            # Ensure doc loader available even if components not fully initialized
            from cubo.ingestion.document_loader import DocumentLoader

            self.doc_loader = DocumentLoader()
        documents = self._load_all_documents(folder)
        return len(documents)

    def ingest_documents_async(self, data_folder: str = None) -> str:
        """
        Start full ingestion (load + remove + index) in background.

        Returns:
            job_id: ID to track status via get_task_status()
        """
        from cubo.processing.background_manager import bg_manager

        # We submit build_index as the task since it does end-to-end ingestion
        return bg_manager.submit_task(self.build_index, data_folder)

    def get_task_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background task."""
        from cubo.processing.background_manager import bg_manager

        return bg_manager.get_status(job_id)

    def query_retrieve(
        self, query: str, top_k: int = None, trace_id: Optional[str] = None, 
        collection_id: Optional[str] = None, doc_ids: Optional[List[str]] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question
            top_k: Number of results to return. Defaults to config value.
            trace_id: Optional trace ID for logging
            collection_id: Optional collection to restrict retrieval to
            doc_ids: Optional list of document IDs to restrict retrieval to
            **kwargs: Additional arguments passed to retriever

        Returns:
            List of retrieved document chunks with metadata.

        Thread-safe: Uses _state_lock.
        """
        with self._state_lock:
            if top_k is None:
                top_k = config.get("retrieval.default_top_k", 6)
            return self.retriever.retrieve_top_documents(
                query, top_k, trace_id=trace_id, 
                collection_id=collection_id, doc_ids=doc_ids, **kwargs
            )

    def generate_response_safe(
        self, query: str, context: str, trace_id: Optional[str] = None
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            query: User's question
            context: Document context to include
            trace_id: Optional trace ID for logging

        Returns:
            Generated response string.

        Thread-safe: Uses _state_lock.
        """
        with self._state_lock:
            return self.generator.generate_response(query=query, context=context, trace_id=trace_id)

    def generate_response_stream(
        self, query: str, context: str, trace_id: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate a streaming response using the LLM.

        Args:
            query: User's question
            context: Document context to include
            trace_id: Optional trace ID for logging

        Yields:
            NDJSON events (token, done, error).

        Thread-safe: Uses _state_lock.
        """
        with self._state_lock:
            yield from self.generator.generate_response_stream(
                query=query, context=context, trace_id=trace_id
            )

    def query(self, query: str, top_k: int = None) -> str:
        """
        Simple query interface for developers.

        Args:
            query: User's question
            top_k: Number of context documents to retrieve

        Returns:
            Generated answer string.
        """
        result = self.query_and_generate(query, top_k)
        return result["answer"]

    def query_and_generate(
        self, query: str, top_k: int = None, trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combined query and response generation.

        Args:
            query: User's question
            top_k: Number of context documents to retrieve
            trace_id: Optional trace ID

        Returns:
            Dict with 'answer', 'sources', and 'trace_id' keys.
        """
        # Retrieve
        docs = self.query_retrieve(query, top_k, trace_id)

        # Build context
        context = "\n\n".join([doc.get("document", doc.get("content", "")) for doc in docs])

        # Generate
        answer = self.generate_response_safe(query, context, trace_id)

        return {"answer": answer, "sources": docs, "trace_id": trace_id}

    def add_documents(self, documents: List[Dict]) -> bool:
        """
        Add documents directly to the vector store.

        Args:
            documents: List of document dicts with 'text' and optionally 'file_path'

        Returns:
            True if any documents were added.
        """
        with self._state_lock:
            if not self.retriever:
                raise RuntimeError("Retriever not initialized. Call initialize_components() first.")
            return self.retriever.add_documents(documents)

    def _load_all_documents(self, data_folder: str) -> List[Dict]:
        """Load and chunk all documents from the data folder."""
        logger.info("Loading and chunking all documents...")
        start = time.time()
        documents = self.doc_loader.load_documents_from_folder(data_folder)
        logger.info(
            f"Documents loaded and chunked into {len(documents)} chunks "
            f"in {time.time() - start:.2f} seconds."
        )
        return documents

    def _add_documents_to_db(self, documents: List[Dict]) -> None:
        """Add documents to the vector database."""
        self.retriever.add_documents(documents)

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down CuboCore...")
        # Future: close connections, flush caches, etc.
        pass


# Convenience factory function
def create_cubo_core() -> CuboCore:
    """
    Factory function to create and optionally initialize a CuboCore instance.

    Returns:
        Initialized CuboCore instance.
    """
    return CuboCore()
