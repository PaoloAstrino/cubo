"""
Document Store Service - Manages document storage and retrieval.

This module handles document lifecycle operations including:
- Adding documents and their chunks to the vector store
- Session tracking for loaded documents
- Chunk preparation and ID generation
- Embedding generation coordination
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from cubo.config import config
from cubo.utils.exceptions import (
    DatabaseError,
    DocumentAlreadyExistsError,
    EmbeddingGenerationError,
    FileAccessError,
    ModelNotAvailableError,
)
from cubo.utils.logger import logger


class DocumentStore:
    """
    Manages document storage operations for the retrieval system.

    Responsibilities:
    - Track which documents are loaded in the current session
    - Prepare chunks with metadata for indexing
    - Generate unique chunk IDs
    - Coordinate embedding generation
    - Add/remove documents from the vector store
    """

    def __init__(
        self,
        collection: Any,
        model: Any = None,
        inference_threading: Any = None,
        use_sentence_window: bool = True,
        bm25_searcher: Any = None,
    ):
        """
        Initialize the document store.

        Args:
            collection: Vector store collection interface
            model: Embedding model (SentenceTransformer)
            inference_threading: Threading helper for embedding generation
            use_sentence_window: Whether using sentence window chunking
            bm25_searcher: BM25 searcher for keyword statistics
        """
        self.collection = collection
        self.model = model
        self.inference_threading = inference_threading
        self.use_sentence_window = use_sentence_window
        self.bm25_searcher = bm25_searcher

        # Track loaded documents in current session
        self.current_documents: Set[str] = set()

    def get_file_hash(self, filepath: str) -> str:
        """
        Get hash of file content for caching and deduplication.

        Args:
            filepath: Path to the file

        Returns:
            MD5 hash of file content

        Raises:
            FileAccessError: If file cannot be accessed
        """
        try:
            with open(filepath, "rb") as f:
                return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
        except FileNotFoundError:
            raise FileAccessError(filepath, "read", {"reason": "file_not_found"})
        except PermissionError:
            raise FileAccessError(filepath, "read", {"reason": "permission_denied"})
        except OSError as e:
            raise FileAccessError(filepath, "read", {"reason": "os_error", "details": str(e)})

    def get_filename_from_path(self, filepath: str) -> str:
        """Extract filename from path."""
        return Path(filepath).name

    def is_document_loaded(self, filepath: str) -> bool:
        """Check if document is already loaded in current session."""
        filename = self.get_filename_from_path(filepath)
        return filename in self.current_documents

    def get_loaded_documents(self) -> List[str]:
        """Get list of currently loaded document filenames."""
        return list(self.current_documents)

    def clear_session(self) -> None:
        """Clear current session document tracking."""
        self.current_documents.clear()
        logger.info("Cleared current session document tracking")

    def check_document_exists(self, filepath: str) -> bool:
        """
        Check if document is already loaded in current session.

        Args:
            filepath: Full path to the document

        Returns:
            True if already exists
        """
        filename = self.get_filename_from_path(filepath)
        if filename in self.current_documents:
            logger.info(f"Document {filename} already loaded in current session")
            return True
        return False

    def check_database_duplicate(self, file_hash: str, filename: str) -> bool:
        """
        Check if document with same hash already exists in database.

        Args:
            file_hash: Hash of the file content
            filename: Just the filename

        Returns:
            True if duplicate exists
        """
        try:
            existing_docs = self.collection.get(where={"file_hash": file_hash})
            if existing_docs.get("ids"):
                logger.info(f"Document {filename} with same content already exists in database")
                # Still add to current session tracking
                self.current_documents.add(filename)
                return True
        except Exception as e:
            logger.warning(f"Error checking for duplicate: {e}")
        return False

    def validate_for_addition(self, filepath: str) -> None:
        """
        Validate that a document can be added.

        Args:
            filepath: Path to the document

        Raises:
            DocumentAlreadyExistsError: If document already exists
        """
        filename = self.get_filename_from_path(filepath)

        if self.check_document_exists(filepath):
            raise DocumentAlreadyExistsError(filename, {"filepath": filepath})

        file_hash = self.get_file_hash(filepath)
        if self.check_database_duplicate(file_hash, filename):
            raise DocumentAlreadyExistsError(filename, {"file_hash": file_hash})

    def prepare_chunk_data(
        self,
        chunks: List[dict],
        filename: str,
        file_hash: str,
        filepath: str,
    ) -> Dict[str, List]:
        """
        Prepare texts and metadata for chunks.

        Args:
            chunks: List of chunk dictionaries
            filename: Document filename
            file_hash: File content hash
            filepath: Full file path

        Returns:
            Dictionary with 'texts' and 'metadatas' lists
        """
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            text, metadata = self._extract_chunk_info(
                chunk, i, filename, file_hash, filepath
            )
            texts.append(text)
            metadatas.append(metadata)

        return {"texts": texts, "metadatas": metadatas}

    def _extract_chunk_info(
        self,
        chunk: Any,
        index: int,
        filename: str,
        file_hash: str,
        filepath: str,
    ) -> tuple:
        """Extract text and metadata from a chunk."""
        if isinstance(chunk, dict):
            return self._extract_sentence_window_chunk(
                chunk, index, filename, file_hash, filepath
            )
        else:
            return self._extract_character_chunk(
                chunk, index, filename, file_hash, filepath
            )

    def _extract_sentence_window_chunk(
        self,
        chunk: dict,
        index: int,
        filename: str,
        file_hash: str,
        filepath: str,
    ) -> tuple:
        """Extract text and metadata from a sentence window chunk."""
        text = chunk["text"]
        metadata = {
            "filename": filename,
            "file_hash": file_hash,
            "filepath": filepath,
            "chunk_index": index,
            "sentence_index": chunk.get("sentence_index", index),
            "window": chunk.get("window", ""),
            "window_start": chunk.get("window_start", index),
            "window_end": chunk.get("window_end", index),
            "sentence_token_count": chunk.get("sentence_token_count", 0),
            "window_token_count": chunk.get("window_token_count", 0),
        }
        return text, metadata

    def _extract_character_chunk(
        self,
        chunk: str,
        index: int,
        filename: str,
        file_hash: str,
        filepath: str,
    ) -> tuple:
        """Extract text and metadata from a character-based chunk."""
        text = chunk
        metadata = {
            "filename": filename,
            "file_hash": file_hash,
            "filepath": filepath,
            "chunk_index": index,
            "token_count": len(chunk.split()),  # Approximate
        }
        return text, metadata

    def create_chunk_ids(
        self,
        metadatas: List[dict],
        filename: str,
    ) -> List[str]:
        """
        Create deterministic IDs for chunks.

        Args:
            metadatas: List of metadata dictionaries
            filename: Document filename

        Returns:
            List of unique IDs for each chunk
        """
        chunk_ids = []
        prefer_hash = config.get("deep_chunk_id_use_file_hash", True)

        for i, metadata in enumerate(metadatas):
            base = None
            if prefer_hash and metadata.get("file_hash"):
                base = metadata.get("file_hash")
            else:
                base = filename

            if self.use_sentence_window and "sentence_index" in metadata:
                sentence_idx = metadata["sentence_index"]
                chunk_ids.append(f"{base}_s{sentence_idx}")
            else:
                chunk_ids.append(f"{base}_chunk_{i}")

        return chunk_ids

    def generate_embeddings(
        self,
        texts: List[str],
        filename: str,
    ) -> List[List[float]]:
        """
        Generate embeddings for text chunks.

        Args:
            texts: List of text chunks
            filename: Document filename for logging

        Returns:
            List of embeddings

        Raises:
            EmbeddingGenerationError: If embedding generation fails
            ModelNotAvailableError: If model not available
        """
        if self.model is None:
            raise ModelNotAvailableError("embedding_model")

        if self.inference_threading is None:
            raise EmbeddingGenerationError(
                "Inference threading not configured",
                {"filename": filename}
            )

        try:
            logger.info(f"Generating embeddings for {filename} ({len(texts)} chunks)")
            embeddings = self.inference_threading.generate_embeddings_threaded(
                texts, self.model
            )

            if not embeddings or len(embeddings) != len(texts):
                raise EmbeddingGenerationError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}",
                    {"filename": filename, "expected_count": len(texts)},
                )

            return embeddings

        except EmbeddingGenerationError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate embeddings for {filename}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(
                error_msg, {"filename": filename, "text_count": len(texts)}
            ) from e

    def add_chunks(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict],
        chunk_ids: List[str],
        filename: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Add chunks to the vector store.

        Args:
            embeddings: List of embeddings
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            chunk_ids: List of unique chunk IDs
            filename: Document filename for logging
            trace_id: Optional trace ID for debugging
        """
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=chunk_ids,
            trace_id=trace_id,
        )

        # Track as loaded in current session
        self.current_documents.add(filename)

        # Update BM25 statistics for keyword search
        if self.bm25_searcher:
            self._update_bm25_statistics(texts, chunk_ids)

        logger.info(f"Successfully added {filename} with {len(chunk_ids)} chunks")

    def _update_bm25_statistics(self, texts: List[str], doc_ids: List[str]) -> None:
        """Update BM25 statistics when documents are added."""
        docs = []
        for doc_id, text in zip(doc_ids, texts):
            docs.append({"doc_id": doc_id, "text": text})

        self.bm25_searcher.add_documents(docs)
        logger.debug(f"Updated BM25 stats: {len(doc_ids)} chunks added.")

    def add_document(
        self,
        filepath: str,
        chunks: List[dict],
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Add document chunks to the database with metadata.

        Args:
            filepath: Path to the document
            chunks: List of chunk dicts
            trace_id: Optional trace ID for debugging

        Returns:
            True if added successfully

        Raises:
            DocumentAlreadyExistsError: If document already exists
            Various other exceptions for other errors
        """
        filename = self.get_filename_from_path(filepath)
        file_hash = self.get_file_hash(filepath)

        self.validate_for_addition(filepath)

        chunk_data = self.prepare_chunk_data(chunks, filename, file_hash, filepath)
        embeddings = self.generate_embeddings(chunk_data["texts"], filename)
        chunk_ids = self.create_chunk_ids(chunk_data["metadatas"], filename)

        self.add_chunks(
            embeddings=embeddings,
            texts=chunk_data["texts"],
            metadatas=chunk_data["metadatas"],
            chunk_ids=chunk_ids,
            filename=filename,
            trace_id=trace_id,
        )

        return True

    def add_test_document(
        self,
        fake_path: str,
        content: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Add a test document directly with content.

        Args:
            fake_path: Fake path for the document
            content: Document content
            trace_id: Optional trace ID

        Returns:
            True if added, False if already exists
        """
        filename = self.get_filename_from_path(fake_path)

        if filename in self.current_documents:
            logger.info(f"Document {filename} already loaded in current session")
            return False

        file_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

        try:
            existing_docs = self.collection.get(where={"file_hash": file_hash})
            if existing_docs.get("ids"):
                logger.info(f"Document {filename} with same content already exists")
                self.current_documents.add(filename)
                return False
        except Exception:
            pass

        embeddings = self.generate_embeddings([content], filename)
        chunk_ids = [f"{filename}_chunk_0"]
        metadatas = [{
            "filename": filename,
            "file_hash": file_hash,
            "filepath": fake_path,
            "chunk_index": 0,
            "total_chunks": 1,
        }]

        self.add_chunks(
            embeddings=embeddings,
            texts=[content],
            metadatas=metadatas,
            chunk_ids=chunk_ids,
            filename=filename,
            trace_id=trace_id,
        )

        return True

    def remove_document(self, filepath: str) -> bool:
        """
        Remove document from current session tracking.
        Note: Chunks remain in database for caching.

        Args:
            filepath: Path to the document

        Returns:
            True if removed from current session
        """
        try:
            filename = self.get_filename_from_path(filepath)
            if filename in self.current_documents:
                self.current_documents.remove(filename)
                logger.info(f"Removed {filename} from current session")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing document {filepath}: {e}")
            return False

    def count_chunks(self) -> int:
        """Get total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def has_documents(self) -> bool:
        """Check if any documents are available for retrieval."""
        if self.current_documents:
            return True

        try:
            result = self.collection.count()
            if result > 0:
                logger.info(
                    f"No session documents, but found {result} chunks in database"
                )
                return True
        except Exception as e:
            logger.error(f"Error checking database: {e}")

        return False

    def debug_collection_info(self) -> Dict:
        """Get debug information about the collection."""
        try:
            count = self.collection.count()
            all_metadata = self.collection.get(include=["metadatas"])

            doc_counts = {}
            if all_metadata.get("metadatas"):
                for metadata in all_metadata["metadatas"]:
                    filename = metadata.get("filename", "unknown")
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1

            return {
                "total_chunks": count,
                "current_session_docs": len(self.current_documents),
                "current_session_filenames": list(self.current_documents),
                "all_documents_in_db": doc_counts,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
