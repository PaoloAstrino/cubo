"""
CUBO Document Retriever
Handles document embedding, storage, and retrieval with ChromaDB.
"""

from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import os
from pathlib import Path
import json
from src.logger import logger
from src.config import config
from src.service_manager import get_service_manager
from src.model_inference_threading import get_model_inference_threading
from .reranker import LocalReranker


class DocumentRetriever:
    """Handles document retrieval using ChromaDB and sentence transformers."""

    def __init__(self, model: SentenceTransformer, use_sentence_window: bool = True,
                 use_auto_merging: bool = False, auto_merge_for_complex: bool = True,
                 window_size: int = 3, top_k: int = 3):
        self.model = model
        self.service_manager = get_service_manager()
        self.inference_threading = get_model_inference_threading()
        self.use_sentence_window = use_sentence_window
        self.use_auto_merging = use_auto_merging
        self.auto_merge_for_complex = auto_merge_for_complex
        self.window_size = window_size
        self.top_k = top_k

        # Initialize auto-merging retriever if enabled
        self.auto_merging_retriever = None
        if self.use_auto_merging:
            try:
                from .custom_auto_merging import AutoMergingRetriever
                self.auto_merging_retriever = AutoMergingRetriever(self.model)
                logger.info("Custom auto-merging retriever initialized")
            except ImportError as e:
                logger.warning(f"Auto-merging retrieval not available: {e}")
                self.use_auto_merging = False

        self.client = chromadb.PersistentClient(
            path=config.get("chroma_db_path", "./chroma_db")
        )
        self.collection = self.client.get_or_create_collection(
            name=config.get("collection_name", "cubo_documents")
        )

        # Track currently loaded documents
        self.current_documents = set()

        # Query cache for testing
        self.query_cache = {}

        # Cache file path for testing
        self.cache_file = os.path.join(config.get("cache_dir", "./cache"), "query_cache.json")
        self._load_cache()  # Load existing cache from disk

        # Initialize postprocessors if using sentence windows
        if self.use_sentence_window:
            from .postprocessor import WindowReplacementPostProcessor
            self.window_postprocessor = WindowReplacementPostProcessor()
            if self.model:
                self.reranker = LocalReranker(self.model)
            else:
                self.reranker = None
                logger.warning("Embedding model not available, reranker will not be initialized.")
        else:
            self.window_postprocessor = None
            self.reranker = None

        logger.info(f"Document retriever initialized (sentence_window={use_sentence_window}, auto_merging={use_auto_merging})")

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

    def add_document(self, filepath: str, chunks: List[dict]) -> bool:
        """
        Add document chunks to the database with metadata.

        Args:
            filepath: Path to the document
            chunks: List of chunk dicts (from sentence window or character chunking)

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
            existing_docs = self.collection.get(where={"file_hash": file_hash})

            if existing_docs.get('ids'):
                logger.info(f"Document {filename} with same content already exists in database")
                # Still add to current session tracking
                self.current_documents.add(filename)
                return False

            # Prepare texts for embedding (use 'text' field for sentence window, or chunk itself for character)
            texts = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    # Sentence window chunk
                    text = chunk["text"]
                    metadata = {
                        "filename": filename,
                        "file_hash": file_hash,
                        "filepath": filepath,
                        "chunk_index": i,
                        "sentence_index": chunk.get("sentence_index", i),
                        "window": chunk.get("window", ""),
                        "window_start": chunk.get("window_start", i),
                        "window_end": chunk.get("window_end", i),
                        "sentence_token_count": chunk.get("sentence_token_count", 0),
                        "window_token_count": chunk.get("window_token_count", 0)
                    }
                else:
                    # Character-based chunk (backward compatibility)
                    text = chunk
                    metadata = {
                        "filename": filename,
                        "file_hash": file_hash,
                        "filepath": filepath,
                        "chunk_index": i,
                        "token_count": len(chunk.split())  # Approximate
                    }

                texts.append(text)
                metadatas.append(metadata)

            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {filename} ({len(texts)} chunks)")
            embeddings = self.inference_threading.generate_embeddings_threaded(texts, self.model)

            # Create deterministic IDs
            chunk_ids = []
            for i, metadata in enumerate(metadatas):
                if self.use_sentence_window and "sentence_index" in metadata:
                    # Use sentence-based ID for sentence windows
                    sentence_idx = metadata["sentence_index"]
                    chunk_ids.append(f"{filename}_s{sentence_idx}")
                else:
                    # Use chunk index for character-based
                    chunk_ids.append(f"{filename}_chunk_{i}")

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=chunk_ids
            )

            # Track as loaded in current session
            self.current_documents.add(filename)

            logger.info(f"Successfully added {filename} with {len(chunk_ids)} chunks")
            return True

        return self.service_manager.execute_sync('document_processing', _add_document_operation)

    def add_documents(self, documents: List[str]) -> bool:
        """
        Add multiple documents directly (for testing purposes).

        Args:
            documents: List of document strings

        Returns:
            bool: True if any documents were added
        """
        if not documents:
            return True

        # For testing, treat each document as a single chunk without requiring files
        added_any = False
        for i, doc in enumerate(documents):
            # Create a fake filepath for testing
            fake_path = f"test_doc_{i}.txt"

            def _add_test_document_operation():
                filename = self._get_filename_from_path(fake_path)

                # Check if document is already loaded in current session
                if self.is_document_loaded(fake_path):
                    logger.info(f"Document {filename} already loaded in current session")
                    return False

                # Use document content hash instead of file hash for testing
                import hashlib
                file_hash = hashlib.md5(doc.encode()).hexdigest()

                # Check if document with same hash already exists
                existing_docs = self.collection.get(where={"file_hash": file_hash})

                if existing_docs['ids']:
                    logger.info(f"Document {filename} with same content already exists in database")
                    # Still add to current session tracking
                    self.current_documents.add(filename)
                    return False

                # Generate embeddings for chunks
                logger.info(f"Generating embeddings for {filename} (1 chunk)")
                embeddings = self.inference_threading.generate_embeddings_threaded([doc], self.model)

                # Create IDs and metadata
                chunk_ids = [f"{filename}_chunk_0"]
                metadatas = [{
                    "filename": filename,
                    "file_hash": file_hash,
                    "filepath": fake_path,
                    "chunk_index": 0,
                    "total_chunks": 1
                }]

                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=[doc],
                    metadatas=metadatas,
                    ids=chunk_ids
                )

                # Track as loaded in current session
                self.current_documents.add(filename)

                logger.info(f"Successfully added {filename} with 1 chunk")
                return True

            success = self.service_manager.execute_sync('document_processing', _add_test_document_operation)
            if success:
                added_any = True

        return added_any

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

    def retrieve_top_documents(self, query: str, top_k: int = 6) -> List[Dict]:
        """
        Retrieve top-k most relevant document chunks for the current session.
        Smart selection between sentence window and auto-merging retrieval.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with document, metadata, and similarity
        """
        # Analyze query complexity to choose retrieval method
        is_complex = self._analyze_query_complexity(query)

        if self.auto_merge_for_complex and is_complex and self.use_auto_merging and self.auto_merging_retriever:
            # Use auto-merging for complex queries
            logger.info("Using auto-merging retrieval for complex query")
            return self._retrieve_auto_merging(query, top_k)
        else:
            # Use sentence window retrieval
            logger.info("Using sentence window retrieval")
            return self._retrieve_sentence_window(query, top_k)

    def _analyze_query_complexity(self, query: str) -> bool:
        """Determine if query needs complex retrieval."""
        complex_indicators = [
            'why', 'how', 'explain', 'compare', 'analyze',
            'relationship', 'difference', 'benefits', 'impact',
            'advantages', 'disadvantages', 'vs', 'versus'
        ]

        query_lower = query.lower()
        # Check for complex keywords
        has_complex_keywords = any(indicator in query_lower for indicator in complex_indicators)
        # Check for long queries
        is_long_query = len(query.split()) > 12

        return has_complex_keywords or is_long_query

    def _retrieve_sentence_window(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using sentence window method."""

        def _retrieve_operation():
            if not self.current_documents:
                logger.warning("No documents loaded in current session")
                return []

            # Generate query embedding
            query_embeddings = self.inference_threading.generate_embeddings_threaded([query], self.model)
            query_embedding = query_embeddings[0] if query_embeddings else []

            # Get more candidates for reranking if using sentence windows
            initial_top_k = min(top_k * 2, 10) if self.use_sentence_window else top_k

            # Query only chunks from currently loaded documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_top_k,
                where={"filename": {"$in": list(self.current_documents)}},  # Only current docs
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            candidates = []
            if results['documents'] and results['metadatas'] and results['distances']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    candidates.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance  # Convert distance to similarity
                    })

            # Apply postprocessing for sentence windows
            if self.use_sentence_window and self.window_postprocessor:
                candidates = self.window_postprocessor.postprocess_results(candidates)

            # Apply reranking if available
            if self.use_sentence_window and self.reranker and len(candidates) > top_k:
                # Rerank but keep all candidates (don't limit to reranker.top_n)
                reranked = self.reranker.rerank(query, candidates, max_results=len(candidates))
                if reranked:
                    candidates = reranked

            # Return top results
            final_results = candidates[:top_k]
            logger.info(f"Retrieved {len(final_results)} chunks using sentence window")
            return final_results

        return self.service_manager.execute_sync('database_operation', _retrieve_operation)

    def _retrieve_auto_merging(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using auto-merging method."""
        try:
            if not self.auto_merging_retriever:
                logger.warning("Auto-merging retriever not available, falling back to sentence window")
                return self._retrieve_sentence_window(query, top_k)

            # Get results from auto-merging retriever
            results = self.auto_merging_retriever.retrieve(query, top_k=top_k)

            # Convert to CUBO format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "document": result.get('document', ''),
                    "metadata": result.get('metadata', {}),
                    "similarity": result.get('similarity', 1.0)
                })

            logger.info(f"Retrieved {len(formatted_results)} chunks using auto-merging")
            return formatted_results

        except Exception as e:
            logger.error(f"Auto-merging retrieval failed: {e}, falling back to sentence window")
            return self._retrieve_sentence_window(query, top_k)

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

    def _save_cache(self):
        """Save query cache to disk (for testing)."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        # Convert tuple keys to strings for JSON serialization
        serializable_cache = {str(k): v for k, v in self.query_cache.items()}
        with open(self.cache_file, 'w') as f:
            json.dump(serializable_cache, f)

    def _load_cache(self):
        """Load query cache from disk (for testing)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    loaded_cache = json.load(f)
                # Convert string keys back to tuples
                self.query_cache = {}
                for k, v in loaded_cache.items():
                    # Parse tuple from string like "('test', 3)"
                    if k.startswith("(") and k.endswith(")"):
                        # Simple parsing for tuple keys
                        parts = k[1:-1].split(", ")
                        if len(parts) == 2:
                            key = (parts[0].strip("'\""), int(parts[1]))
                            self.query_cache[key] = v
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.query_cache = {}
