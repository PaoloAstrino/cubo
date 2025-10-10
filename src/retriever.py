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
import math
import re
from collections import Counter, defaultdict
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
        
        # BM25 parameters and document statistics
        self.bm25_k1 = 1.5  # Term frequency saturation parameter
        self.bm25_b = 0.75  # Length normalization parameter
        self.doc_lengths = {}  # Document ID -> length
        self.avg_doc_length = 0
        self.term_doc_freq = defaultdict(int)  # Term -> number of docs containing it
        self.doc_term_freq = {}  # Doc ID -> {term: frequency}

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
            # If both retrieval methods are available, use hybrid approach
            if self.use_auto_merging and self._is_auto_merging_available():
                # Perform both retrieval methods
                sentence_results = self._retrieve_sentence_window(
                    query, top_k // 2 + top_k % 2
                )  # Slightly more for sentence window
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
            else:
                # Use only sentence window retrieval
                return self._retrieve_sentence_window(query, top_k)

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
        """
        Retrieve using sentence window method with hybrid semantic + BM25 scoring.
        
        Does two parallel retrievals:
        1. Pure semantic similarity search
        2. Pure BM25 keyword search
        
        Then combines results with 50/50 weighting.
        """

        def _retrieve_operation():
            if not self._has_loaded_documents():
                return []

            query_embedding = self._generate_query_embedding(query)
            
            # Retrieve more candidates for each method
            retrieval_k = top_k * 3
            
            # Method 1: Pure semantic retrieval
            semantic_candidates = self._query_collection_for_candidates(
                query_embedding, retrieval_k, query=""
            )
            
            # Method 2: Pure BM25 retrieval (scan all docs and score by BM25)
            bm25_candidates = self._retrieve_by_bm25(query, retrieval_k)
            
            # Combine with 50/50 weighting
            combined_candidates = self._combine_semantic_and_bm25(
                semantic_candidates, bm25_candidates, top_k
            )
            
            # Apply sentence window postprocessing
            combined_candidates = self._apply_sentence_window_postprocessing(
                combined_candidates, top_k, query
            )

            self._log_retrieval_results(combined_candidates, "sentence window (hybrid)")
            return combined_candidates

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

        # Update BM25 statistics for keyword search
        self._update_bm25_statistics(texts, chunk_ids)

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
        
        Retrieve more candidates to allow BM25 keyword scoring to find
        semantically dissimilar but lexically relevant documents.

        Args:
            top_k: Final number of results desired

        Returns:
            int: Initial number of candidates to retrieve
        """
        # Retrieve 5x more candidates to allow BM25 reranking to work effectively
        return top_k * 5 if self.use_sentence_window else top_k

    def _query_collection_for_candidates(
        self, query_embedding: List[float], initial_top_k: int, query: str = ""
    ) -> List[dict]:
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

                # Apply keyword boost with detailed breakdown
                score_breakdown = self._apply_keyword_boost_detailed(doc, query, base_similarity)

                # Update metadata with score breakdown for tracking
                updated_metadata = metadata.copy()
                updated_metadata['score_breakdown'] = score_breakdown

                candidates.append({
                    "document": doc,
                    "metadata": updated_metadata,
                    "similarity": score_breakdown["final_score"],
                    "base_similarity": base_similarity  # Keep original for debugging
                })
        return candidates

    def _update_bm25_statistics(self, texts: List[str], doc_ids: List[str]) -> None:
        """
        Update BM25 statistics when documents are added.

        Args:
            texts: List of document texts
            doc_ids: List of document IDs
        """
        for doc_id, text in zip(doc_ids, texts):
            # Tokenize document
            tokens = self._tokenize(text)
            
            # Store term frequencies for this document
            term_freq = Counter(tokens)
            self.doc_term_freq[doc_id] = term_freq
            self.doc_lengths[doc_id] = len(tokens)
            
            # Update document frequency for each unique term
            for term in term_freq.keys():
                self.term_doc_freq[term] += 1
        
        # Update average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        
        logger.debug(f"Updated BM25 stats: {len(doc_ids)} chunks, "
                     f"total chunks={len(self.doc_lengths)}, avg_len={self.avg_doc_length:.1f}, "
                     f"unique_terms={len(self.term_doc_freq)}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing punctuation and lowercasing."""
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stop words
        stop_words = {'tell', 'me', 'about', 'the', 'what', 'is', 'a', 'an', 'and', 'or',
                      'describe', 'explain', 'how', 'why', 'when', 'where', 'who', 'which',
                      'that', 'this', 'these', 'those', 'was', 'were', 'are', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'of', 'at', 'by', 'for', 'with', 'from', 'to', 'in', 'on'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _compute_bm25_score(self, query_terms: List[str], doc_id: str, doc_text: str) -> float:
        """
        Compute BM25 score for a document given query terms.

        BM25 formula:
        score = sum over each query term t of:
            IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

        where:
        - IDF(t) = log((N - df + 0.5) / (df + 0.5))
        - tf = term frequency in document
        - doc_len = document length
        - N = total number of documents
        - df = document frequency of term
        """
        if doc_id not in self.doc_term_freq:
            # Build term frequencies for this document on-the-fly
            tokens = self._tokenize(doc_text)
            self.doc_term_freq[doc_id] = Counter(tokens)
            self.doc_lengths[doc_id] = len(tokens)

        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            return 0.0

        # Get total number of documents
        total_docs = len(self.doc_lengths) if self.doc_lengths else 1
        avg_len = self.avg_doc_length if self.avg_doc_length > 0 else doc_len

        score = 0.0
        doc_terms = self.doc_term_freq.get(doc_id, {})

        for term in query_terms:
            # Term frequency in this document
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue

            # Document frequency (how many docs contain this term)
            df = self.term_doc_freq.get(term, 0)

            # IDF calculation
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score
            numerator = tf * (self.bm25_k1 + 1)
            denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (doc_len / avg_len))
            term_score = idf * (numerator / denominator)

            score += term_score

        return score

    def _apply_keyword_boost(self, document: str, query: str, base_similarity: float) -> float:
        """
        Boost similarity score using BM25 keyword scoring.

        Args:
            document: Document text
            query: Query text
            base_similarity: Original semantic similarity score

        Returns:
            Combined score (weighted average of semantic and BM25)
        """
        if not query or not document:
            return base_similarity

        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return base_similarity

        # Compute BM25 score (use a pseudo doc_id based on document hash)
        doc_id = hashlib.md5(document.encode(), usedforsecurity=False).hexdigest()[:8]
        bm25_score = self._compute_bm25_score(query_terms, doc_id, document)

        # Normalize BM25 score to [0, 1] range (rough approximation)
        # BM25 scores typically range from 0 to ~10-20 for relevant docs
        normalized_bm25 = min(bm25_score / 15.0, 1.0)

        # Apply BM25 boost ONLY when there's a keyword match
        # This avoids penalizing documents with no keyword matches
        if normalized_bm25 > 0.05:
            # Add up to 30% boost based on BM25 score
            boost_factor = 0.3 * normalized_bm25
            combined_score = base_similarity + boost_factor
            # Cap at 1.0 to keep in valid similarity range
            combined_score = min(combined_score, 1.0)
            
            logger.debug(f"BM25 BOOST: raw={bm25_score:.3f}, norm={normalized_bm25:.3f}, "
                         f"semantic={base_similarity:.3f}, boost={boost_factor:.3f}, combined={combined_score:.3f}")
            return combined_score
        else:
            # No meaningful keyword match, return original semantic similarity
            return base_similarity

    def _apply_keyword_boost_detailed(self, document: str, query: str, base_similarity: float) -> Dict[str, float]:
        """
        Boost similarity score using BM25 keyword scoring and return detailed breakdown.

        Args:
            document: Document text
            query: Query text
            base_similarity: Original semantic similarity score

        Returns:
            Dict with final_score, semantic_score, bm25_score, semantic_contribution, bm25_contribution
        """
        if not query or not document:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0
            }

        # Tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0
            }

        # Compute BM25 score (use a pseudo doc_id based on document hash)
        doc_id = hashlib.md5(document.encode(), usedforsecurity=False).hexdigest()[:8]
        bm25_score = self._compute_bm25_score(query_terms, doc_id, document)

        # Normalize BM25 score to [0, 1] range (rough approximation)
        # BM25 scores typically range from 0 to ~10-20 for relevant docs
        normalized_bm25 = min(bm25_score / 15.0, 1.0)

        # Calculate weighted contributions: 10% semantic + 90% BM25
        semantic_weight = 0.1
        bm25_weight = 0.9

        semantic_contribution = semantic_weight * base_similarity
        bm25_contribution = bm25_weight * normalized_bm25
        final_score = semantic_contribution + bm25_contribution

        # Cap at 1.0 to keep in valid similarity range
        final_score = min(final_score, 1.0)

        logger.debug(f"DETAILED SCORE: semantic={base_similarity:.3f} (contrib={semantic_contribution:.3f}), "
                     f"bm25_raw={bm25_score:.3f}, bm25_norm={normalized_bm25:.3f} (contrib={bm25_contribution:.3f}), "
                     f"final={final_score:.3f}")

        return {
            "final_score": final_score,
            "semantic_score": base_similarity,
            "bm25_score": bm25_score,
            "semantic_contribution": semantic_contribution,
            "bm25_contribution": bm25_contribution
        }

    def _retrieve_by_bm25(self, query: str, top_k: int) -> List[dict]:
        """
        Retrieve documents using pure BM25 scoring.
        
        Scans all documents in the collection and ranks by BM25 score only.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents ranked by BM25 score
        """
        # Get all documents from collection
        try:
            all_docs = self.collection.get(
                include=['documents', 'metadatas'],
                where={"filename": {"$in": list(self.current_documents)}} if self.current_documents else None
            )
            
            if not all_docs['documents']:
                return []
            
            # Tokenize query
            query_terms = self._tokenize(query)
            if not query_terms:
                return []
            
            # Score all documents by BM25
            scored_docs = []
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc_id = hashlib.md5(doc.encode(), usedforsecurity=False).hexdigest()[:8]
                bm25_score = self._compute_bm25_score(query_terms, doc_id, doc)
                
                # Normalize to [0, 1]
                normalized_score = min(bm25_score / 15.0, 1.0)
                
                if normalized_score > 0.01:  # Only include docs with some keyword match
                    scored_docs.append({
                        "document": doc,
                        "metadata": metadata,
                        "similarity": normalized_score,
                        "base_similarity": 0.0,  # No semantic score for pure BM25
                        "bm25_score": bm25_score
                    })
            
            # Sort by BM25 score and return top_k
            scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"BM25 retrieval: scored {len(scored_docs)} docs, returning top {min(top_k, len(scored_docs))}")
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []

    def _combine_semantic_and_bm25(
        self, semantic_candidates: List[dict], bm25_candidates: List[dict], top_k: int
    ) -> List[dict]:
        """
        Combine semantic and BM25 retrieval results with 50/50 weighting.
        
        Args:
            semantic_candidates: Results from semantic similarity search
            bm25_candidates: Results from BM25 keyword search
            top_k: Number of final results to return
            
        Returns:
            Combined and re-ranked results
        """
        # Create a document index for deduplication
        combined = {}
        
        # Add semantic results (50% weight)
        for cand in semantic_candidates:
            doc_key = cand['document'][:100]  # Use first 100 chars as key
            if doc_key not in combined:
                combined[doc_key] = {
                    "document": cand['document'],
                    "metadata": cand['metadata'],
                    "semantic_score": cand.get('base_similarity', cand['similarity']),
                    "bm25_score": 0.0
                }
            else:
                # Update semantic score if we have a better one
                combined[doc_key]['semantic_score'] = max(
                    combined[doc_key]['semantic_score'],
                    cand.get('base_similarity', cand['similarity'])
                )
        
        # Add BM25 results (50% weight)
        for cand in bm25_candidates:
            doc_key = cand['document'][:100]
            if doc_key not in combined:
                combined[doc_key] = {
                    "document": cand['document'],
                    "metadata": cand['metadata'],
                    "semantic_score": 0.0,
                    "bm25_score": cand['similarity']
                }
            else:
                # Update BM25 score
                combined[doc_key]['bm25_score'] = max(
                    combined[doc_key]['bm25_score'],
                    cand['similarity']
                )
        
        # Compute combined scores (10% semantic + 90% BM25)
        # Give strong preference to keyword matches for entity-specific retrieval
        final_results = []
        for doc_data in combined.values():
            combined_score = 0.1 * doc_data['semantic_score'] + 0.9 * doc_data['bm25_score']
            final_results.append({
                "document": doc_data['document'],
                "metadata": doc_data['metadata'],
                "similarity": combined_score,
                "base_similarity": doc_data['semantic_score'],
                "bm25_normalized": doc_data['bm25_score']
            })
            
            # Debug logging (commented out for production)
            # if doc_data['bm25_score'] > 0.05 or doc_data['semantic_score'] > 0.4:
            #     filename = doc_data['metadata'].get('filename', 'unknown')
            #     logger.info(f"  {filename}: sem={doc_data['semantic_score']:.3f}, "
            #                f"bm25={doc_data['bm25_score']:.3f}, combined={combined_score:.3f}")
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Combined {len(semantic_candidates)} semantic + {len(bm25_candidates)} BM25 "
                   f"results into {len(final_results)} unique docs")
        
        return final_results[:top_k]

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
