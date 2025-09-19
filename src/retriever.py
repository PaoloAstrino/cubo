"""
CUBO Document Retriever
Handles document embedding, storage, and retrieval with ChromaDB.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
import os
from pathlib import Path
import json
from src.logger import logger
from src.config import config
from src.service_manager import get_service_manager


class DocumentRetriever:
    """Handles document retrieval using ChromaDB and sentence transformers."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.service_manager = get_service_manager()
        
        self.client = chromadb.PersistentClient(
            path=config.get("chroma_db_path", "./chroma_db")
        )
        self.collection = self.client.get_or_create_collection(
            name=config.get("collection_name", "cubo_documents")
        )

        # Track currently loaded documents
        self.current_documents = set()

        logger.info("Document retriever initialized")

    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for caching."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_filename_from_path(self, filepath: str) -> str:
        """Extract filename from path."""
        return Path(filepath).name

    def is_document_loaded(self, filepath: str) -> bool:
        """Check if document is already loaded in current session."""
        filename = self._get_filename_from_path(filepath)
        return filename in self.current_documents

    def add_document(self, filepath: str, chunks: List[str]) -> bool:
        """
        Add document chunks to the database with metadata.

        Args:
            filepath: Path to the document
            chunks: List of text chunks

        Returns:
            bool: True if added, False if already exists
        """
        def _add_document_operation():
            filename = self._get_filename_from_path(filepath)

            # Check if document is already loaded in current session
            if self.is_document_loaded(filepath):
                logger.info(f"Document {filename} already loaded in current session")
                return False

            # Get file hash for caching
            file_hash = self._get_file_hash(filepath)

            # Check if document with same hash already exists
            existing_docs = self.service_manager.execute_sync(
                'database_operation',
                lambda: self.collection.get(where={"file_hash": file_hash})
            )

            if existing_docs['ids']:
                logger.info(f"Document {filename} with same content already exists in database")
                # Still add to current session tracking
                self.current_documents.add(filename)
                return False

            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {filename} ({len(chunks)} chunks)")
            embeddings = self.service_manager.execute_sync(
                'embedding_generation',
                lambda: self.model.encode(chunks).tolist()
            )

            # Create IDs and metadata
            chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{
                "filename": filename,
                "file_hash": file_hash,
                "filepath": filepath,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]

            # Add to collection
            self.service_manager.execute_sync(
                'database_operation',
                lambda: self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
            )

            # Track as loaded in current session
            self.current_documents.add(filename)

            logger.info(f"Successfully added {filename} with {len(chunks)} chunks")
            return True

        return self.service_manager.execute_sync('document_processing', _add_document_operation)

    def remove_document(self, filepath: str) -> bool:
        """
        Remove document from current session tracking.
        Note: Chunks remain in database for caching.

        Args:
            filepath: Path to the document

        Returns:
            bool: True if removed from current session
        """
        try:
            filename = self._get_filename_from_path(filepath)
            if filename in self.current_documents:
                self.current_documents.remove(filename)
                logger.info(f"Removed {filename} from current session")
                return True
            return False

        except Exception as e:
            logger.error(f"Error removing document {filepath}: {e}")
            return False

    def retrieve_top_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant document chunks for the current session.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with document, metadata, and similarity
        """
        def _retrieve_operation():
            if not self.current_documents:
                logger.warning("No documents loaded in current session")
                return []

            # Generate query embedding
            query_embedding = self.service_manager.execute_sync(
                'embedding_generation',
                lambda: self.model.encode([query]).tolist()[0]
            )

            # Query only chunks from currently loaded documents
            results = self.service_manager.execute_sync(
                'database_operation',
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where={"filename": {"$in": list(self.current_documents)}},  # Only current docs
                    include=['documents', 'metadatas', 'distances']
                )
            )

            # Format results
            top_docs = []
            if results['documents'] and results['metadatas'] and results['distances']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    top_docs.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance  # Convert distance to similarity
                    })

            logger.info(f"Retrieved {len(top_docs)} chunks from {len(self.current_documents)} current documents")
            return top_docs

        return self.service_manager.execute_sync('database_operation', _retrieve_operation)

    def get_loaded_documents(self) -> List[str]:
        """Get list of currently loaded document filenames."""
        return list(self.current_documents)

    def clear_current_session(self):
        """Clear current session document tracking."""
        self.current_documents.clear()
        logger.info("Cleared current session document tracking")

    def debug_collection_info(self) -> Dict:
        """Get debug information about the collection."""
        try:
            count = self.collection.count()
            all_metadata = self.collection.get(include=['metadatas'])

            # Count documents by filename
            doc_counts = {}
            if all_metadata.get('metadatas'):
                for metadata in all_metadata['metadatas']:
                    filename = metadata.get('filename', 'unknown')
                    doc_counts[filename] = doc_counts.get(filename, 0) + 1

            return {
                "total_chunks": count,
                "current_session_docs": len(self.current_documents),
                "current_session_filenames": list(self.current_documents),
                "all_documents_in_db": doc_counts
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
