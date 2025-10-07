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
from src.exceptions import (
    CUBOError, DatabaseError, DocumentAlreadyExistsError, EmbeddingGenerationError,
    ModelNotAvailableError, FileAccessError, RetrievalError
)

from .reranker import LocalReranker


class DocumentRetriever:
    """Handles document retrieval using ChromaDB and sentence transformers."""

    def __init__(self, model: SentenceTransformer, use_sentence_window: bool = True,
                 use_auto_merging: bool = False, auto_merge_for_complex: bool = True,
                 window_size: int = 3, top_k: int = 3):
        self._set_basic_attributes(model, use_sentence_window, use_auto_merging,
                                   auto_merge_for_complex, window_size, top_k)
        self._initialize_auto_merging_retriever()
        self._setup_chromadb()
        self._setup_caching()
        self._initialize_postprocessors()
        self._log_initialization_status()

    def _set_basic_attributes(self, model: SentenceTransformer, use_sentence_window: bool,
                              use_auto_merging: bool, auto_merge_for_complex: bool,
                              window_size: int, top_k: int) -> None:
        """Set basic instance attributes."""
        self.model = model
        self.service_manager = get_service_manager()
        self.inference_threading = get_model_inference_threading()
        self.use_sentence_window = use_sentence_window
        self.use_auto_merging = use_auto_merging
        self.auto_merge_for_complex = auto_merge_for_complex
        self.window_size = window_size
        self.top_k = top_k

    def _initialize_auto_merging_retriever(self) -> None:
        """Initialize auto-merging retriever if enabled."""
        self.auto_merging_retriever = None
        if self.use_auto_merging:
            try:
                from .custom_auto_merging import AutoMergingRetriever
                self.auto_merging_retriever = AutoMergingRetriever(self.model)
                logger.info("Custom auto-merging retriever initialized")
            except ImportError as e:
                logger.warning(f"Auto-merging retrieval not available: {e}")
                self.use_auto_merging = False

    def _setup_chromadb(self) -> None:
        """Setup ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=config.get("chroma_db_path", "./chroma_db")
        )
        self.collection = self.client.get_or_create_collection(
            name=config.get("collection_name", "cubo_documents")
        )

    def _setup_caching(self) -> None:
        """Setup query caching for testing."""
        self.current_documents = set()
        self.query_cache = {}
        self.cache_file = os.path.join(config.get("cache_dir", "./cache"), "query_cache.json")
        self._load_cache()

    def _initialize_postprocessors(self) -> None:
        """Initialize postprocessors and reranker based on configuration."""
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

    def _log_initialization_status(self) -> None:
        """Log the initialization status."""
        logger.info(f"Document retriever initialized "
                    f"(sentence_window={self.use_sentence_window}, "
                    f"auto_merging={self.use_auto_merging})")

    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for caching."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read(), usedforsecurity=False).hexdigest()
        except FileNotFoundError:
            raise FileAccessError(filepath, "read", {"reason": "file_not_found"})
        except PermissionError:
            raise FileAccessError(filepath, "read", {"reason": "permission_denied"})
        except OSError as e:
            raise FileAccessError(filepath, "read", {"reason": "os_error", "details": str(e)})

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

        Raises:
            FileAccessError: If file cannot be accessed
            DatabaseError: If database operation fails
            EmbeddingGenerationError: If embedding generation fails
        """
        try:
            return self.service_manager.execute_sync('document_processing',
                                                     lambda: self._add_document_operation(filepath, chunks))
        except DocumentAlreadyExistsError:
            # Document already exists - this is not an error, just return False
            logger.info(f"Document {self._get_filename_from_path(filepath)} already exists")
            return False
        except CUBOError:
            # Re-raise other custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            error_msg = f"Unexpected error adding document {filepath}: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg, "ADD_DOCUMENT_FAILED", {"filepath": filepath}) from e

    def _add_document_operation(self, filepath: str, chunks: List[dict]) -> bool:
        """Execute the document addition operation."""
        filename = self._get_filename_from_path(filepath)
        self._validate_document_for_addition(filepath, filename)

        chunk_data = self._prepare_chunk_data(chunks, filename,
                                              self._get_file_hash(filepath), filepath)
        success = self._process_and_add_document(chunk_data, filename)

        # Also add to auto-merging retriever if available
        if success and self.auto_merging_retriever:
            try:
                auto_merge_success = self.auto_merging_retriever.add_document(filepath)
                if auto_merge_success:
                    logger.info(f"Document {filename} also added to auto-merging retriever")
                else:
                    logger.warning(f"Failed to add document {filename} to auto-merging retriever")
            except Exception as e:
                logger.error(f"Error adding document {filename} to auto-merging retriever: {e}")

        return success

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

            success = self._add_test_document(fake_path, doc)
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
        Retrieve top-k most relevant document chunks using hybrid retrieval.
        Combines sentence window and auto-merging for better coverage.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with document, metadata, and similarity

        Raises:
            RetrievalMethodUnavailableError: If no retrieval methods are available
            RetrievalError: If retrieval operation fails
        """
        try:
            # Perform both retrieval methods
            sentence_results = self._retrieve_sentence_window(query, top_k // 2 + top_k % 2)  # Slightly more for sentence window
            auto_results = self._retrieve_auto_merging_safe(query, top_k // 2)

            # Combine results
            combined_results = sentence_results + auto_results

            # Deduplicate by document content
            seen_content = set()
            unique_results = []
            for result in combined_results:
                # Use 'document' field for deduplication
                content = result.get('document', result.get('content', ''))
                if content not in seen_content:
                    seen_content.add(content)
                    unique_results.append(result)

            # Re-sort by similarity score after deduplication
            # This ensures best matches across all documents come first
            unique_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
            
            # Log the source distribution for debugging
            source_files = [r.get('metadata', {}).get('filename', 'Unknown') for r in unique_results[:top_k]]
            logger.info(f"Hybrid retrieval returning results from: {source_files}")

            # Return top_k unique results
            return unique_results[:top_k]

        except CUBOError:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error during document retrieval: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(error_msg, "RETRIEVAL_FAILED", {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "top_k": top_k
            }) from e

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
            if not self._has_loaded_documents():
                return []

            query_embedding = self._generate_query_embedding(query)
            initial_top_k = self._calculate_initial_top_k(top_k)
            candidates = self._query_collection_for_candidates(query_embedding, initial_top_k, query)
            candidates = self._apply_sentence_window_postprocessing(candidates, top_k, query)

            self._log_retrieval_results(candidates, "sentence window")
            return candidates

        return self.service_manager.execute_sync('database_operation', _retrieve_operation)

    def _has_loaded_documents(self) -> bool:
        """Check if any documents are available for retrieval."""
        # If we have documents in current session, use them
        if self.current_documents:
            return True
        
        # Otherwise, check if there are ANY documents in the database
        try:
            result = self.collection.count()
            if result > 0:
                logger.info(f"No session documents, but found {result} chunks in database - allowing retrieval")
                return True
        except Exception as e:
            logger.error(f"Error checking database: {e}")
        
        logger.warning("No documents available for retrieval")
        return False

    def _log_retrieval_results(self, candidates: List[Dict], method: str):
        """Log the results of a retrieval operation."""
        logger.info(f"Retrieved {len(candidates)} chunks using {method}")

    def _retrieve_auto_merging(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using auto-merging method."""
        try:
            if not self._is_auto_merging_available():
                return self._fallback_to_sentence_window(query, top_k)

            results = self.auto_merging_retriever.retrieve(query, top_k=top_k)
            formatted_results = self._format_auto_merging_results(results)

            self._log_retrieval_results(formatted_results, "auto-merging")
            return formatted_results

        except Exception as e:
            return self._handle_auto_merging_error(e, query, top_k)

    def _is_auto_merging_available(self) -> bool:
        """Check if auto-merging retriever is available."""
        return self.auto_merging_retriever is not None

    def _fallback_to_sentence_window(self, query: str, top_k: int) -> List[Dict]:
        """Fallback to sentence window retrieval."""
        logger.warning("Auto-merging retriever not available, falling back to sentence window")
        return self._retrieve_sentence_window(query, top_k)

    def _format_auto_merging_results(self, results) -> List[Dict]:
        """Convert auto-merging results to CUBO format."""
        formatted_results = []
        for result in results:
            formatted_results.append({
                "document": result.get('document', ''),
                "metadata": result.get('metadata', {}),
                "similarity": result.get('similarity', 1.0)
            })
        return formatted_results

    def _handle_auto_merging_error(self, error: Exception, query: str, top_k: int) -> List[Dict]:
        """Handle auto-merging retrieval errors."""
        logger.error(f"Auto-merging retrieval failed: {error}, falling back to sentence window")
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

    def _check_document_exists(self, filepath: str, filename: str) -> bool:
        """
        Check if document is already loaded in current session.

        Args:
            filepath: Full path to the document
            filename: Just the filename

        Returns:
            bool: True if already exists
        """
        if self.is_document_loaded(filepath):
            logger.info(f"Document {filename} already loaded in current session")
            return True
        return False

    def _check_database_duplicate(self, file_hash: str, filename: str) -> bool:
        """
        Check if document with same hash already exists in database.

        Args:
            file_hash: Hash of the file content
            filename: Just the filename

        Returns:
            bool: True if duplicate exists
        """
        existing_docs = self.collection.get(where={"file_hash": file_hash})
        if existing_docs.get('ids'):
            logger.info(f"Document {filename} with same content already exists in database")
            # Still add to current session tracking
            self.current_documents.add(filename)
            return True
        return False

    def _prepare_chunk_data(self, chunks: List[dict], filename: str, file_hash: str, filepath: str) -> dict:
        """
        Prepare texts and metadata for chunks.

        Args:
            chunks: List of chunk dictionaries
            filename: Document filename
            file_hash: File content hash
            filepath: Full file path

        Returns:
            dict: Dictionary with 'texts' and 'metadatas' lists
        """
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            text, metadata = self._extract_chunk_info(chunk, i, filename, file_hash, filepath)
            texts.append(text)
            metadatas.append(metadata)

        return {"texts": texts, "metadatas": metadatas}

    def _extract_chunk_info(self, chunk, index: int, filename: str, file_hash: str, filepath: str):
        """Extract text and metadata from a chunk."""
        if isinstance(chunk, dict):
            return self._extract_sentence_window_chunk(chunk, index, filename, file_hash, filepath)
        else:
            return self._extract_character_chunk(chunk, index, filename, file_hash, filepath)

    def _extract_sentence_window_chunk(self, chunk: dict, index: int, filename: str, file_hash: str, filepath: str):
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
            "window_token_count": chunk.get("window_token_count", 0)
        }
        return text, metadata

    def _extract_character_chunk(self, chunk: str, index: int, filename: str, file_hash: str, filepath: str):
        """Extract text and metadata from a character-based chunk."""
        text = chunk
        metadata = {
            "filename": filename,
            "file_hash": file_hash,
            "filepath": filepath,
            "chunk_index": index,
            "token_count": len(chunk.split())  # Approximate
        }
        return text, metadata

    def _generate_chunk_embeddings(self, texts: List[str], filename: str) -> List[List[float]]:
        """
        Generate embeddings for text chunks.

        Args:
            texts: List of text chunks
            filename: Document filename for logging

        Returns:
            List[List[float]]: Embeddings for each chunk

        Raises:
            EmbeddingGenerationError: If embedding generation fails
            ModelNotAvailableError: If the model is not available
        """
        self._validate_model_availability()

        try:
            logger.info(f"Generating embeddings for {filename} ({len(texts)} chunks)")
            embeddings = self.inference_threading.generate_embeddings_threaded(texts, self.model)

            self._validate_embeddings_result(embeddings, texts, filename)

            return embeddings

        except Exception as e:
            # Re-raise with more context
            error_msg = f"Failed to generate embeddings for {filename}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg, {"filename": filename, "text_count": len(texts)}) from e

    def _validate_model_availability(self) -> None:
        """
        Validate that the embedding model is available.

        Raises:
            ModelNotAvailableError: If the model is not available
        """
        if not self.model:
            raise ModelNotAvailableError("embedding_model")

    def _validate_embeddings_result(self, embeddings: List[List[float]], texts: List[str], filename: str) -> None:
        """
        Validate the embeddings generation result.

        Args:
            embeddings: Generated embeddings
            texts: Original texts
            filename: Document filename for error context

        Raises:
            EmbeddingGenerationError: If validation fails
        """
        if not embeddings or len(embeddings) != len(texts):
            raise EmbeddingGenerationError(
                f"Expected {len(texts)} embeddings, got {len(embeddings) if embeddings else 0}",
                {"filename": filename, "expected_count": len(texts), "actual_count": len(embeddings) if embeddings else 0}
            )

    def _create_chunk_ids(self, metadatas: List[dict], filename: str) -> List[str]:
        """
        Create deterministic IDs for chunks.

        Args:
            metadatas: List of metadata dictionaries
            filename: Document filename

        Returns:
            List[str]: Unique IDs for each chunk
        """
        chunk_ids = []
        for i, metadata in enumerate(metadatas):
            if self.use_sentence_window and "sentence_index" in metadata:
                # Use sentence-based ID for sentence windows
                sentence_idx = metadata["sentence_index"]
                chunk_ids.append(f"{filename}_s{sentence_idx}")
            else:
                # Use chunk index for character-based
                chunk_ids.append(f"{filename}_chunk_{i}")
        return chunk_ids

    def _add_chunks_to_collection(self, embeddings: List[List[float]],
                                  texts: List[str], metadatas: List[Dict],
                                  chunk_ids: List[str], filename: str) -> None:
        """
        Add chunks to the ChromaDB collection.

        Args:
            embeddings: List of embeddings
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            chunk_ids: List of unique chunk IDs
            filename: Document filename for logging
        """
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=chunk_ids
        )

        # Track as loaded in current session
        self.current_documents.add(filename)

        logger.info(f"Successfully added {filename} with {len(chunk_ids)} chunks")

    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            query: Search query

        Returns:
            List[float]: Query embedding vector
        """
        query_embeddings = self.inference_threading.generate_embeddings_threaded([query], self.model)
        return query_embeddings[0] if query_embeddings else []

    def _calculate_initial_top_k(self, top_k: int) -> int:
        """
        Calculate initial number of candidates to retrieve before reranking.

        Args:
            top_k: Final number of results desired

        Returns:
            int: Initial number of candidates to retrieve
        """
        return min(top_k * 2, 10) if self.use_sentence_window else top_k

    def _query_collection_for_candidates(self, query_embedding: List[float], initial_top_k: int, query: str = "") -> List[dict]:
        """
        Query the collection for candidate documents.

        Args:
            query_embedding: Query embedding vector
            initial_top_k: Number of candidates to retrieve
            query: Original query text for keyword boosting

        Returns:
            List[dict]: Candidate documents with metadata

        Raises:
            DatabaseError: If database query fails
        """
        try:
            results = self._execute_collection_query(query_embedding, initial_top_k)
            return self._process_query_results(results, query)
        except Exception as e:
            error_msg = f"Failed to query document collection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg, "QUERY_FAILED", {
                "query_embedding_length": len(query_embedding),
                "top_k": initial_top_k,
                "current_docs_count": len(self.current_documents)
            }) from e

    def _execute_collection_query(self, query_embedding: List[float], initial_top_k: int):
        """Execute the ChromaDB collection query."""
        # If no documents in current session, search ALL documents in database
        # Otherwise, only search current session documents
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": initial_top_k,
            "include": ['documents', 'metadatas', 'distances']
        }
        
        if self.current_documents:
            query_params["where"] = {"filename": {"$in": list(self.current_documents)}}
            logger.debug(f"Searching in session documents: {self.current_documents}")
        else:
            logger.debug("No session filter - searching all documents in database")
        
        return self.collection.query(**query_params)

    def _process_query_results(self, results, query: str = "") -> List[dict]:
        """Process raw query results into candidate format with optional keyword boosting."""
        candidates = []
        if results['documents'] and results['metadatas'] and results['distances']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                base_similarity = 1 - distance  # Convert distance to similarity
                
                # Apply keyword boost if query contains specific terms
                boosted_similarity = self._apply_keyword_boost(doc, query, base_similarity)
                
                candidates.append({
                    "document": doc,
                    "metadata": metadata,
                    "similarity": boosted_similarity,
                    "base_similarity": base_similarity  # Keep original for debugging
                })
        return candidates
    
    def _apply_keyword_boost(self, document: str, query: str, base_similarity: float) -> float:
        """
        Boost similarity score if document contains important keywords from query.
        
        Args:
            document: Document text
            query: Query text
            base_similarity: Original similarity score
            
        Returns:
            Boosted similarity score
        """
        if not query:
            return base_similarity
            
        # Extract important words from query (remove common words)
        stop_words = {'tell', 'me', 'about', 'the', 'what', 'is', 'a', 'an', 'describe', 'explain', 'how', 'why'}
        query_words = [word.lower().strip('.,!?') for word in query.split() if word.lower() not in stop_words]
        
        if not query_words:
            return base_similarity
        
        # Check if any important query words appear in the document
        doc_lower = document.lower()
        matches = sum(1 for word in query_words if word in doc_lower)
        
        if matches > 0:
            # Boost by 0.3 * (proportion of query words found)
            boost = 0.3 * (matches / len(query_words))
            boosted = min(base_similarity + boost, 1.0)  # Cap at 1.0
            logger.debug(f"Keyword boost: {matches}/{len(query_words)} words matched, "
                        f"similarity {base_similarity:.4f} -> {boosted:.4f}")
            return boosted
        
        return base_similarity

    def _apply_sentence_window_postprocessing(self, candidates: List[dict], top_k: int, query: str) -> List[dict]:
        """
        Apply postprocessing for sentence window retrieval.

        Args:
            candidates: Raw candidate documents
            top_k: Final number of results to return

        Returns:
            List[dict]: Processed and reranked candidates
        """
        # Apply postprocessing for sentence windows
        candidates = self._apply_window_postprocessing(candidates)

        # Apply reranking if available
        candidates = self._apply_reranking_if_available(candidates, top_k, query)

        # Return top results
        return candidates[:top_k]

    def _apply_window_postprocessing(self, candidates: List[dict]) -> List[dict]:
        """
        Apply window postprocessing if available.

        Args:
            candidates: Raw candidate documents

        Returns:
            List[dict]: Postprocessed candidates
        """
        if self.use_sentence_window and self.window_postprocessor:
            return self.window_postprocessor.postprocess_results(candidates)
        return candidates

    def _apply_reranking_if_available(self, candidates: List[dict], top_k: int, query: str) -> List[dict]:
        """
        Apply reranking if available and beneficial.

        Args:
            candidates: Candidate documents
            top_k: Number of final results needed
            query: Search query

        Returns:
            List[dict]: Reranked candidates if reranking was applied
        """
        if self.use_sentence_window and self.reranker and len(candidates) > top_k:
            # Rerank but keep all candidates (don't limit to reranker.top_n)
            reranked = self.reranker.rerank(query, candidates, max_results=len(candidates))
            if reranked:
                return reranked
        return candidates

    def _add_test_document(self, fake_path: str, doc: str) -> bool:
        """
        Add a single test document to the collection.

        Args:
            fake_path: Fake filepath for the test document
            doc: Document content

        Returns:
            bool: True if document was added successfully
        """
        def _add_test_document_operation():
            filename = self._get_filename_from_path(fake_path)

            # Check if document is already loaded in current session
            if self._check_test_document_loaded(fake_path, filename):
                return False

            # Generate content hash
            file_hash = self._generate_content_hash(doc)

            # Check if document with same hash already exists
            if self._check_test_document_duplicate(file_hash, filename):
                return False

            # Prepare and add document data
            return self._prepare_and_add_test_document(doc, filename, file_hash, fake_path)

        return self.service_manager.execute_sync('document_processing', _add_test_document_operation)

    def _check_test_document_loaded(self, fake_path: str, filename: str) -> bool:
        """
        Check if test document is already loaded in current session.

        Args:
            fake_path: Fake filepath for the test document
            filename: Document filename

        Returns:
            bool: True if already loaded
        """
        if self.is_document_loaded(fake_path):
            logger.info(f"Document {filename} already loaded in current session")
            return True
        return False

    def _generate_content_hash(self, doc: str) -> str:
        """
        Generate MD5 hash from document content.

        Args:
            doc: Document content

        Returns:
            str: MD5 hash of the content
        """
        import hashlib
        return hashlib.md5(doc.encode(), usedforsecurity=False).hexdigest()

    def _check_test_document_duplicate(self, file_hash: str, filename: str) -> bool:
        """
        Check if document with same hash already exists in database.

        Args:
            file_hash: Content hash of the document
            filename: Document filename

        Returns:
            bool: True if duplicate exists
        """
        existing_docs = self.collection.get(where={"file_hash": file_hash})

        if existing_docs['ids']:
            logger.info(f"Document {filename} with same content already exists in database")
            # Still add to current session tracking
            self.current_documents.add(filename)
            return True
        return False

    def _prepare_and_add_test_document(self, doc: str, filename: str, file_hash: str, fake_path: str) -> bool:
        """
        Prepare document data and add to collection.

        Args:
            doc: Document content
            filename: Document filename
            file_hash: Content hash
            fake_path: Fake filepath

        Returns:
            bool: True if successfully added
        """
        embeddings = self._generate_test_embeddings(doc, filename)
        chunk_ids, metadatas = self._create_test_chunk_metadata(doc, filename, file_hash, fake_path)

        self._add_test_chunks_to_collection(embeddings, [doc], metadatas, chunk_ids, filename)
        return True

    def _generate_test_embeddings(self, doc: str, filename: str) -> List[List[float]]:
        """Generate embeddings for test document."""
        logger.info(f"Generating embeddings for {filename} (1 chunk)")
        return self.inference_threading.generate_embeddings_threaded([doc], self.model)

    def _create_test_chunk_metadata(self, doc: str, filename: str, file_hash: str, fake_path: str):
        """Create chunk IDs and metadata for test document."""
        chunk_ids = [f"{filename}_chunk_0"]
        metadatas = [{
            "filename": filename,
            "file_hash": file_hash,
            "filepath": fake_path,
            "chunk_index": 0,
            "total_chunks": 1
        }]
        return chunk_ids, metadatas

    def _add_test_chunks_to_collection(self, embeddings: List[List[float]],
                                       documents: List[str], metadatas: List[Dict],
                                       chunk_ids: List[str], filename: str) -> None:
        """Add test chunks to the collection."""
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=chunk_ids
        )

        # Track as loaded in current session
        self.current_documents.add(filename)
        logger.info(f"Successfully added {filename} with 1 chunk")

    def _choose_retrieval_method(self, query: str) -> str:
        """
        Choose the appropriate retrieval method based on query complexity.

        Args:
            query: Search query

        Returns:
            str: 'auto_merging' or 'sentence_window'
        """
        is_complex = self._analyze_query_complexity(query)

        if self.auto_merge_for_complex and is_complex and self.use_auto_merging and self.auto_merging_retriever:
            return 'auto_merging'
        else:
            return 'sentence_window'

    def _execute_retrieval(self, method: str, query: str, top_k: int) -> List[Dict]:
        """
        Execute the chosen retrieval method with fallback handling.

        Args:
            method: Retrieval method ('auto_merging' or 'sentence_window')
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results
        """
        if method == 'auto_merging':
            logger.info("Using auto-merging retrieval for complex query")
            try:
                return self._retrieve_auto_merging_safe(query, top_k)
            except Exception as e:
                logger.error(f"Auto-merging retrieval failed: {e}, falling back to sentence window")
                return self._retrieve_sentence_window(query, top_k)
        else:
            logger.info("Using sentence window retrieval")
            return self._retrieve_sentence_window(query, top_k)

    def _retrieve_auto_merging_safe(self, query: str, top_k: int) -> List[Dict]:
        """
        Safely retrieve using auto-merging method with fallback check.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieval results

        Raises:
            Exception: If auto-merging retriever is not available
        """
        if not self.auto_merging_retriever:
            raise Exception("Auto-merging retriever not available")

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

    def _validate_document_for_addition(self, filepath: str, filename: str) -> None:
        """
        Validate that a document can be added.

        Args:
            filepath: Path to the document
            filename: Document filename

        Raises:
            DocumentAlreadyExistsError: If document already exists
        """
        # Check if document is already loaded
        if self._check_document_exists(filepath, filename):
            raise DocumentAlreadyExistsError(filename, {"filepath": filepath})

        # Get file hash for caching
        file_hash = self._get_file_hash(filepath)

        # Check if document with same hash already exists in database
        if self._check_database_duplicate(file_hash, filename):
            raise DocumentAlreadyExistsError(filename, {"file_hash": file_hash})

    def _process_and_add_document(self, chunk_data: dict, filename: str) -> bool:
        """
        Process and add document data to the collection.

        Args:
            chunk_data: Prepared chunk data
            filename: Document filename

        Returns:
            bool: True if successfully added
        """
        # Generate embeddings
        embeddings = self._generate_chunk_embeddings(chunk_data['texts'], filename)

        # Create chunk IDs
        chunk_ids = self._create_chunk_ids(chunk_data['metadatas'], filename)

        # Add to collection
        self._add_chunks_to_collection(embeddings, chunk_data['texts'], chunk_data['metadatas'], chunk_ids, filename)

        return True
