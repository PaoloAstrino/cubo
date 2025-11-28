"""
CUBO Document Retriever
Handles document embedding, storage, and retrieval with FAISS.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64

# Lazy imports to avoid loading large native dependencies during test collection
import hashlib

from sentence_transformers import SentenceTransformer

from src.cubo.config import config
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
from src.cubo.embeddings.model_inference_threading import get_model_inference_threading
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.rerank.reranker import LocalReranker
from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.retrieval.cache import SemanticCache
from src.cubo.retrieval.constants import (
    BM25_NORMALIZATION_FACTOR,
    BM25_WEIGHT_DETAILED,
    COMPLEXITY_LENGTH_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_WINDOW_SIZE,
    INITIAL_RETRIEVAL_MULTIPLIER,
    KEYWORD_BOOST_FACTOR,
    MIN_BM25_THRESHOLD,
    SEMANTIC_WEIGHT_DETAILED,
)
from src.cubo.retrieval.fusion import rrf_fuse
from src.cubo.retrieval.orchestrator import (
    DeduplicationManager,
    HybridScorer,
    RetrievalOrchestrator,
    TieredRetrievalManager,
)
from src.cubo.retrieval.strategy import RetrievalStrategy
from src.cubo.services.service_manager import get_service_manager
from src.cubo.storage.memory_store import InMemoryCollection
from src.cubo.utils.exceptions import (
    CUBOError,
    DatabaseError,
    DocumentAlreadyExistsError,
    EmbeddingGenerationError,
    FileAccessError,
    ModelNotAvailableError,
    RetrievalError,
)
from src.cubo.utils.logger import logger
from src.cubo.utils.trace_collector import trace_collector


class DocumentRetriever:
    """Handles document retrieval using FAISS and sentence transformers."""

    """
    Note: `DocumentRetriever` will attempt to initialize a reranker based on the
    `retrieval.reranker_model` configuration. If a cross-encoder model is available
    and configured, the class initializes a `CrossEncoderReranker` as `self.reranker`.
    If CrossEncoder is not available, or the model cannot be loaded, it falls back
    to `LocalReranker` which uses the embedding generator for scoring. If no model
    is configured or loaded, `self.reranker` will be `None` and re-ranking will be
    skipped.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        use_sentence_window: bool = True,
        use_auto_merging: bool = False,
        auto_merge_for_complex: bool = True,
        window_size: int = DEFAULT_WINDOW_SIZE,
        top_k: int = DEFAULT_TOP_K,
    ):
        self._set_basic_attributes(
            model, use_sentence_window, use_auto_merging, auto_merge_for_complex, window_size, top_k
        )
        self._initialize_auto_merging_retriever()
        self._setup_vector_store()
        self._setup_caching()
        self._load_dedup_metadata()
        self._initialize_postprocessors()
        self._initialize_retrieval_strategy()
        self._initialize_tiered_retrieval()
        self._log_initialization_status()

    def close(self) -> None:
        """Close the retriever and release any underlying resources such as
        vector store executors and caches.

        This should be used in production and tests to deterministically free
        resources instead of relying on destructors or garbage collection.
        """
        # Save any cache to disk
        try:
            self._save_cache()
        except Exception:
            pass

        # Close or reset the collection
        try:
            close_fn = getattr(self.collection, "close", None)
            if callable(close_fn):
                close_fn()
            else:
                reset_fn = getattr(self.collection, "reset", None)
                if callable(reset_fn):
                    reset_fn()
        except Exception:
            pass

        # Mark closed flag to prevent further operations
        try:
            self._closed = True
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _set_basic_attributes(
        self,
        model: SentenceTransformer,
        use_sentence_window: bool,
        use_auto_merging: bool,
        auto_merge_for_complex: bool,
        window_size: int,
        top_k: int,
    ) -> None:
        """Set basic instance attributes."""
        self.model = model
        self.service_manager = get_service_manager()
        self.inference_threading = get_model_inference_threading()
        self.use_sentence_window = use_sentence_window
        self.use_auto_merging = use_auto_merging
        self.auto_merge_for_complex = auto_merge_for_complex
        self.window_size = window_size
        self.top_k = top_k
        self.dedup_enabled = bool(config.get("deduplication.enabled", False))
        self.dedup_cluster_lookup: Dict[str, int] = {}
        self.dedup_representatives: Dict[int, Dict[str, Any]] = {}
        self.dedup_canonical_lookup: Dict[str, str] = {}
        self._dedup_map_loaded = False
        # Initialize router for query strategy selection
        try:
            from src.cubo.retrieval.router import SemanticRouter

            self.router = SemanticRouter()
        except Exception:
            # Router initialization failure should not break retriever
            self.router = None

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

    def _setup_vector_store(self) -> None:
        """Setup FAISS vector store.

        If FAISS is unavailable, create a simple in-memory fallback
        collection so unit tests can run without dependencies.
        """
        # Prefer configured backend; default is FAISS for local desktop-focused usage
        from src.cubo.retrieval.vector_store import create_vector_store

        backend = config.get("vector_store_backend", "faiss")

        # Get actual dimension from the loaded model, or use default for tests
        if self.model is not None:
            model_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Using embedding dimension: {model_dimension} (from model)")
        else:
            # Fallback for tests without a model - use common dimension
            model_dimension = 384  # Standard dimension for many sentence transformers
            logger.warning("No model available, using default dimension 384 for vector store initialization")

        try:
            try:
                self.collection = create_vector_store(
                    backend=backend,
                    dimension=model_dimension,
                    index_dir=config.get("vector_store_path"),
                    collection_name=config.get("collection_name", "cubo_documents"),
                )
                return
            except Exception as e:
                logger.warning(
                    f"Vector store initialization failed for backend {backend}: {e}. Falling back to in-memory collection."
                )
        except Exception as e:
            logger.warning(
                f"Vector store initialization error: {e}. Using in-memory fallback collection."
            )

        # Use InMemoryCollection fallback
        self.collection = InMemoryCollection()

    def _setup_caching(self) -> None:
        """Setup query caching for testing."""
        self.current_documents = set()
        self.query_cache = {}
        self.cache_file = os.path.join(config.get("cache_dir", "./cache"), "query_cache.json")
        self._load_cache()  # Load existing cache if available
        self.semantic_cache = None
        cache_enabled = config.get("retrieval.semantic_cache.enabled", False)
        if cache_enabled:
            cache_path = config.get(
                "retrieval.semantic_cache.path",
                os.path.join(config.get("cache_dir", "./cache"), "semantic_cache.json"),
            )
            ttl = int(config.get("retrieval.semantic_cache.ttl", 600))
            threshold = float(config.get("retrieval.semantic_cache.threshold", 0.93))
            max_entries = int(config.get("retrieval.semantic_cache.max_entries", 512))
            self.semantic_cache = SemanticCache(
                ttl_seconds=ttl,
                similarity_threshold=threshold,
                max_entries=max_entries,
                cache_path=cache_path,
            )

        # BM25 searcher initialization
        bm25_stats_path = config.get("bm25_stats_path", "data/bm25_stats.json")
        self.bm25 = BM25Searcher(bm25_stats=bm25_stats_path)
        if bm25_stats_path and os.path.exists(bm25_stats_path):
            logger.info(f"Loaded BM25 stats from {bm25_stats_path}")

    def _load_dedup_metadata(self) -> None:
        """Load deduplication map if configured."""
        if not self.dedup_enabled:
            return
        map_path = config.get("deduplication.map_path")
        if not map_path or not os.path.exists(map_path):
            logger.info("Dedup map not found at %s; disabling deduplication", map_path)
            self.dedup_enabled = False
            return
        try:
            with open(map_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load dedup map %s: %s", map_path, exc)
            self.dedup_enabled = False
            return

        canonical_map = payload.get("canonical_map", {}) or {}
        clusters = payload.get("clusters", {}) or {}
        representatives = payload.get("representatives", {}) or {}
        self.dedup_canonical_lookup = {str(k): str(v) for k, v in canonical_map.items()}
        self.dedup_cluster_lookup.clear()
        for cluster_id, members in clusters.items():
            if not isinstance(members, list):
                continue
            for member in members:
                self.dedup_cluster_lookup[str(member)] = int(cluster_id)
        self.dedup_representatives.clear()
        for cluster_id, rep in representatives.items():
            chunk_id = rep.get("chunk_id")
            if not chunk_id:
                continue
            cid = int(cluster_id)
            self.dedup_representatives[cid] = {
                "chunk_id": str(chunk_id),
                "score": rep.get("score", 0.0),
            }
        self._dedup_map_loaded = True
        logger.info(
            "Loaded dedup map with %s clusters (%s representatives)",
            len(self.dedup_cluster_lookup),
            len(self.dedup_representatives),
        )

    def _initialize_postprocessors(self) -> None:
        """Initialize postprocessors and reranker based on configuration."""
        if self.use_sentence_window:
            from src.cubo.processing.postprocessor import WindowReplacementPostProcessor

            self.window_postprocessor = WindowReplacementPostProcessor()
            # Reranker decision: cross-encoder if configured, otherwise local reranker
            reranker_model = config.get("retrieval.reranker_model", None)
            if reranker_model and isinstance(reranker_model, str):
                try:
                    from src.cubo.rerank.reranker import CrossEncoderReranker

                    self.reranker = CrossEncoderReranker(
                        model_name=reranker_model, top_n=self.top_k
                    )
                except Exception:
                    logger.warning(
                        "Failed to initialize CrossEncoderReranker, falling back to LocalReranker"
                    )
                    self.reranker = LocalReranker(self.model) if self.model else None
            else:
                if self.model:
                    self.reranker = LocalReranker(self.model)
                else:
                    self.reranker = None
                    logger.warning(
                        "Embedding model not available, reranker will not be initialized."
                    )
        else:
            self.window_postprocessor = None
            self.reranker = None

    def _initialize_retrieval_strategy(self) -> None:
        """Initialize retrieval strategy for combining results."""
        self.retrieval_strategy = RetrievalStrategy()

    def _initialize_tiered_retrieval(self) -> None:
        """Initialize scaffold retriever and summary embeddings for three-tier retrieval."""
        # Tiered retrieval configuration
        self.use_summary_prefilter = config.get("retrieval.use_summary_prefilter", False)
        self.use_scaffold_compression = config.get("retrieval.use_scaffold_compression", False)
        self.summary_prefilter_k = config.get("retrieval.summary_prefilter_k", 20)
        self.scaffold_weight = config.get("retrieval.scaffold_weight", 0.3)
        self.summary_weight = config.get("retrieval.summary_weight", 0.2)
        self.dense_weight = config.get("retrieval.dense_weight", 0.5)

        # Scaffold retriever
        self.scaffold_retriever = None
        if self.use_scaffold_compression:
            try:
                from src.cubo.embeddings.embedding_generator import EmbeddingGenerator
                from src.cubo.retrieval.scaffold_retriever import (
                    create_scaffold_retriever_from_directory,
                )

                scaffold_dir = config.get("scaffold.output_dir", "./data/scaffolds")
                if Path(scaffold_dir).exists():
                    embedding_gen = EmbeddingGenerator()
                    self.scaffold_retriever = create_scaffold_retriever_from_directory(
                        scaffold_dir=scaffold_dir, embedding_generator=embedding_gen
                    )
                    if self.scaffold_retriever.is_ready:
                        logger.info(f"Scaffold retriever initialized from {scaffold_dir}")
                    else:
                        logger.warning("Scaffold retriever loaded but not ready (missing data)")
                        self.scaffold_retriever = None
                else:
                    logger.info(
                        f"Scaffold directory not found: {scaffold_dir}, skipping scaffold retrieval"
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize scaffold retriever: {e}")
                self.scaffold_retriever = None

        # Summary embeddings for prefilter
        self.summary_embedder = None
        self.summary_embeddings = None
        self.summary_chunk_ids = None
        if self.use_summary_prefilter:
            try:
                from src.cubo.embeddings.summary_embedder import SummaryEmbedder

                summary_dir = Path(
                    config.get("summary_embeddings.output_dir", "./data/summary_embeddings")
                )
                if summary_dir.exists():
                    self.summary_embedder = SummaryEmbedder()
                    summary_data = self.summary_embedder.load_summary_embeddings(summary_dir)
                    self.summary_embeddings = summary_data.get("embeddings")
                    self.summary_chunk_ids = summary_data.get("chunk_ids", [])
                    if self.summary_embeddings is not None and len(self.summary_embeddings) > 0:
                        logger.info(f"Summary embeddings loaded: {self.summary_embeddings.shape}")
                    else:
                        logger.warning("Summary embeddings loaded but empty")
                        self.summary_embeddings = None
                else:
                    logger.info(
                        f"Summary embeddings directory not found: {summary_dir}, skipping prefilter"
                    )
            except Exception as e:
                logger.warning(f"Failed to load summary embeddings: {e}")
                self.summary_embeddings = None

        # Initialize the orchestrator with specialized managers
        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Initialize the retrieval orchestrator with specialized managers."""
        # Create tiered retrieval manager
        tiered_manager = TieredRetrievalManager(
            summary_embeddings=self.summary_embeddings,
            summary_chunk_ids=self.summary_chunk_ids or [],
            scaffold_retriever=self.scaffold_retriever,
            summary_weight=self.summary_weight,
            scaffold_weight=self.scaffold_weight,
            dense_weight=self.dense_weight,
        )

        # Create hybrid scorer
        hybrid_scorer = HybridScorer()

        # Create deduplication manager
        dedup_manager = DeduplicationManager(
            enabled=self.dedup_enabled,
            cluster_lookup=self.dedup_cluster_lookup,
            representatives=self.dedup_representatives,
            canonical_lookup=self.dedup_canonical_lookup,
        )

        # Create orchestrator
        self.orchestrator = RetrievalOrchestrator(
            tiered_manager=tiered_manager,
            hybrid_scorer=hybrid_scorer,
            dedup_manager=dedup_manager,
        )

    def _log_initialization_status(self) -> None:
        """Log the initialization status."""
        tiered_status = []
        if self.scaffold_retriever:
            tiered_status.append("scaffold")
        if self.summary_embeddings is not None:
            tiered_status.append("summary_prefilter")
        tiered_str = f", tiered=[{','.join(tiered_status)}]" if tiered_status else ""

        logger.info(
            f"Document retriever initialized "
            f"(sentence_window={self.use_sentence_window}, "
            f"auto_merging={self.use_auto_merging}{tiered_str})"
        )

    @property
    def client(self):
        """Return the vector store client if available (for compatibility)."""
        if hasattr(self.collection, "client"):
            return self.collection.client
        return None

    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for caching."""
        try:
            with open(filepath, "rb") as f:
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
            return self.service_manager.execute_sync(
                "document_processing", lambda: self._add_document_operation(filepath, chunks)
            )
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

        chunk_data = self._prepare_chunk_data(
            chunks, filename, self._get_file_hash(filepath), filepath
        )
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

    def add_documents(self, documents: list) -> bool:
        """
        Add multiple documents directly.

        Args:
            documents: List of document strings OR list of chunk dicts with 'text' key

        Returns:
            bool: True if any documents were added
        """
        if not documents:
            return True

        added_any = False
        for i, doc in enumerate(documents):
            # Handle both string and dict formats
            if isinstance(doc, dict):
                text = doc.get("text", "")
                filename = doc.get("filename", f"doc_{i}.txt")
                filepath = doc.get("file_path", f"test_doc_{i}.txt")
            else:
                text = str(doc)
                filename = f"doc_{i}.txt"
                filepath = f"test_doc_{i}.txt"

            if not text:
                continue

            success = self._add_test_document(filepath, text)
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

    def retrieve_top_documents(
        self, query: str, top_k: int = 6, trace_id: Optional[str] = None, **kwargs
    ) -> List[Dict]:
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
            # Backwards compatibility: some callers may pass 'k' instead of 'top_k'
            if kwargs and "k" in kwargs:
                try:
                    top_k = int(kwargs.get("k"))
                except Exception:
                    logger.warning("Invalid 'k' kwarg provided to retrieve_top_documents; ignoring")

            if trace_id:
                try:
                    trace_collector.record(
                        trace_id, "retriever", "start", {"query": query, "top_k": top_k}
                    )
                except Exception:
                    pass
            # Determine retrieval strategy via router (if available)
            if self.router:
                strategy = self.router.route_query(query)
            else:
                strategy = None
            # If both retrieval methods are available, use hybrid approach
            if self.use_auto_merging and self._is_auto_merging_available():
                # Perform both retrieval methods
                sentence_results = self._retrieve_sentence_window(
                    query, top_k // 2 + top_k % 2, strategy=strategy, trace_id=trace_id
                )  # Slightly more for sentence window
                auto_results = self._retrieve_auto_merging_safe(
                    query, top_k // 2, trace_id=trace_id
                )

                # Combine results
                combined_results = sentence_results + auto_results

                unique_results = self._deduplicate_results(combined_results)

                # Re-sort by similarity score after deduplication
                # This ensures best matches across all documents come first
                unique_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

                # Log the source distribution for debugging
                source_files = [
                    r.get("metadata", {}).get("filename", "Unknown") for r in unique_results[:top_k]
                ]
                logger.info(f"Hybrid retrieval returning results from: {source_files}")

                # Return top_k unique results
                results = unique_results[:top_k]
                if trace_id:
                    try:
                        trace_collector.record(
                            trace_id,
                            "retriever",
                            "candidates",
                            {
                                "method": "hybrid",
                                "candidates": [
                                    {
                                        "id": r.get("metadata", {}).get(
                                            "id", r.get("metadata", {}).get("chunk_id", "")
                                        ),
                                        "similarity": r.get("similarity", r.get("score", 0.0)),
                                        "filename": r.get("metadata", {}).get("filename"),
                                    }
                                    for r in results[:10]
                                ],
                            },
                        )
                    except Exception:
                        pass
                return results
            else:
                # Use only sentence window retrieval
                results = self._retrieve_sentence_window(
                    query, top_k, strategy=strategy, trace_id=trace_id
                )
                if trace_id:
                    try:
                        trace_collector.record(
                            trace_id,
                            "retriever",
                            "candidates",
                            {
                                "method": "sentence_window",
                                "candidates": [
                                    {
                                        "id": r.get("metadata", {}).get(
                                            "id", r.get("metadata", {}).get("chunk_id", "")
                                        ),
                                        "similarity": r.get("similarity", r.get("score", 0.0)),
                                        "filename": r.get("metadata", {}).get("filename"),
                                    }
                                    for r in results[:10]
                                ],
                            },
                        )
                    except Exception:
                        pass
                return results

        except CUBOError:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error during document retrieval: {str(e)}"
            logger.error(error_msg)
            raise RetrievalError(
                error_msg,
                "RETRIEVAL_FAILED",
                {"query": query[:100] + "..." if len(query) > 100 else query, "top_k": top_k},
            ) from e

    def _analyze_query_complexity(self, query: str) -> bool:
        """Determine if query needs complex retrieval."""
        # Load complexity heuristics from config with sensible defaults
        complex_indicators = config.get(
            "retrieval.complexity_keywords",
            [
                "why",
                "how",
                "explain",
                "compare",
                "analyze",
                "relationship",
                "difference",
                "benefits",
                "impact",
                "advantages",
                "disadvantages",
                "vs",
                "versus",
            ],
        )
        length_threshold = config.get(
            "retrieval.complexity_length_threshold", COMPLEXITY_LENGTH_THRESHOLD
        )

        query_lower = query.lower()
        # Check for complex keywords
        has_complex_keywords = any(indicator in query_lower for indicator in complex_indicators)
        # Check for long queries
        is_long_query = len(query.split()) > length_threshold

        return has_complex_keywords or is_long_query

    def _retrieve_sentence_window(
        self,
        query: str,
        top_k: int,
        strategy: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve using sentence window method with three-tier retrieval:

        Tier 1: Summary prefilter (fast, broad) - if enabled
        Tier 2: Scaffold compression (semantic grouping) - if enabled
        Tier 3: Dense vectors + BM25 hybrid (precise, focused)

        Then combines results with configurable weighting.
        """

        def _retrieve_operation():
            if not self._has_loaded_documents():
                return []

            query_embedding = self._generate_query_embedding(query)

            # Determine how many candidates to retrieve based on strategy
            if strategy and strategy.get("k_candidates"):
                retrieval_k = int(strategy.get("k_candidates"))
            else:
                retrieval_k = top_k * 3

            # ============================================================
            # TIER 1: Summary Prefilter (lightweight, fast)
            # ============================================================
            summary_chunk_ids = set()
            if self.use_summary_prefilter and self.summary_embeddings is not None:
                try:
                    summary_chunk_ids = self._retrieve_via_summary_prefilter(
                        query_embedding, self.summary_prefilter_k
                    )
                    logger.debug(f"Summary prefilter returned {len(summary_chunk_ids)} chunk IDs")
                except Exception as e:
                    logger.warning(f"Summary prefilter failed: {e}")

            # ============================================================
            # TIER 2: Scaffold Compression (semantic clustering)
            # ============================================================
            scaffold_chunk_ids = set()
            scaffold_scores = {}
            if self.use_scaffold_compression and self.scaffold_retriever:
                try:
                    scaffold_results = self.scaffold_retriever.retrieve_scaffolds(
                        query=query, top_k=max(5, top_k // 2), expand_to_chunks=True
                    )
                    for result in scaffold_results:
                        chunk_ids = result.get("chunk_ids", [])
                        score = result.get("score", 0.5)
                        for cid in chunk_ids:
                            scaffold_chunk_ids.add(cid)
                            scaffold_scores[cid] = max(scaffold_scores.get(cid, 0), score)
                    logger.debug(f"Scaffold retrieval returned {len(scaffold_chunk_ids)} chunk IDs")
                except Exception as e:
                    logger.warning(f"Scaffold retrieval failed: {e}")

            # ============================================================
            # TIER 3: Dense Vector + BM25 Hybrid (standard retrieval)
            # ============================================================
            # Method 1: Pure semantic retrieval
            semantic_candidates = self._query_collection_for_candidates(
                query_embedding, retrieval_k, query="", trace_id=trace_id
            )

            # Method 2: Pure BM25 retrieval (scan all docs and score by BM25)
            bm25_candidates = self._retrieve_by_bm25(query, retrieval_k)

            # Combine semantic and BM25 with weights specified in strategy
            bm25_weight = strategy.get("bm25_weight", 0.3) if strategy else 0.3
            dense_weight = strategy.get("dense_weight", 0.7) if strategy else 0.7
            combined_candidates = self._combine_semantic_and_bm25(
                semantic_candidates,
                bm25_candidates,
                retrieval_k,
                semantic_weight=dense_weight,
                bm25_weight=bm25_weight,
            )

            # ============================================================
            # Combine all tiers with score boosting
            # ============================================================
            combined_candidates = self._apply_tiered_boosting(
                combined_candidates,
                summary_chunk_ids=summary_chunk_ids,
                scaffold_chunk_ids=scaffold_chunk_ids,
                scaffold_scores=scaffold_scores,
            )

            # Apply sentence window postprocessing
            combined_candidates = self._apply_sentence_window_postprocessing(
                combined_candidates, top_k, query, strategy=strategy
            )

            self._log_retrieval_results(combined_candidates, "three-tier hybrid")
            return combined_candidates

        return self.service_manager.execute_sync("database_operation", _retrieve_operation)

    def _retrieve_via_summary_prefilter(self, query_embedding: np.ndarray, k: int) -> Set[str]:
        """
        Use summary embeddings to quickly identify candidate chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of summary matches to return

        Returns:
            Set of chunk IDs that match the query via summary
        """
        if self.summary_embeddings is None or len(self.summary_embeddings) == 0:
            return set()

        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Handle dimension mismatch (e.g., if PCA compression was used)
        if query_embedding.shape[1] != self.summary_embeddings.shape[1]:
            # If summary embeddings are compressed, we need to compress query too
            # For now, skip if dimensions don't match
            logger.debug(
                f"Summary embedding dimension mismatch: query={query_embedding.shape[1]}, summary={self.summary_embeddings.shape[1]}"
            )
            return set()

        # Compute cosine similarity
        query_norm = query_embedding / (
            np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-8
        )
        summary_norm = self.summary_embeddings / (
            np.linalg.norm(self.summary_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = np.dot(query_norm, summary_norm.T).flatten()

        # Get top-k indices
        top_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return chunk IDs
        return {self.summary_chunk_ids[i] for i in top_indices if i < len(self.summary_chunk_ids)}

    def _apply_tiered_boosting(
        self,
        candidates: List[Dict],
        summary_chunk_ids: Set[str],
        scaffold_chunk_ids: Set[str],
        scaffold_scores: Dict[str, float],
    ) -> List[Dict]:
        """
        Apply score boosting based on tiered retrieval matches.

        Chunks that appear in summary prefilter or scaffold results get boosted.

        Args:
            candidates: Base candidates from dense+BM25 retrieval
            summary_chunk_ids: Chunk IDs from summary prefilter
            scaffold_chunk_ids: Chunk IDs from scaffold expansion
            scaffold_scores: Scaffold match scores by chunk ID

        Returns:
            Candidates with boosted scores
        """
        if not summary_chunk_ids and not scaffold_chunk_ids:
            return candidates

        boosted = []
        for candidate in candidates:
            chunk_id = self._extract_chunk_id(candidate)
            base_score = candidate.get("similarity", 0.5)
            boost = 0.0

            # Boost for summary prefilter match
            if chunk_id and chunk_id in summary_chunk_ids:
                boost += self.summary_weight * 0.2  # Small boost for summary match

            # Boost for scaffold match
            if chunk_id and chunk_id in scaffold_chunk_ids:
                scaffold_score = scaffold_scores.get(chunk_id, 0.5)
                boost += self.scaffold_weight * scaffold_score * 0.3

            boosted_candidate = candidate.copy()
            boosted_candidate["similarity"] = min(1.0, base_score + boost)
            if boost > 0:
                boosted_candidate["tier_boost"] = boost
            boosted.append(boosted_candidate)

        # Re-sort by boosted score
        boosted.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

        return boosted

    def _has_loaded_documents(self) -> bool:
        """Check if any documents are available for retrieval."""
        # If we have documents in current session, use them
        if self.current_documents:
            return True

        # Otherwise, check if there are ANY documents in the database
        try:
            result = self.collection.count()
            if result > 0:
                logger.info(
                    f"No session documents, but found {result} chunks in database - allowing retrieval"
                )
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
            formatted_results.append(
                {
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": result.get("similarity", 1.0),
                }
            )
        return formatted_results

    def _handle_auto_merging_error(self, error: Exception, query: str, top_k: int) -> List[Dict]:
        """Handle auto-merging retrieval errors."""
        logger.error(f"Auto-merging retrieval failed: {error}, falling back to sentence window")
        return self._retrieve_sentence_window(query, top_k)

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate retrieval results using cluster metadata when available."""
        if not results:
            return []
        if not self.dedup_enabled or not self._dedup_map_loaded:
            return self._dedup_by_content(results)

        seen: Set[Any] = set()
        deduped: List[Dict] = []
        for result in results:
            chunk_id = self._extract_chunk_id(result)
            cluster_id = self.dedup_cluster_lookup.get(chunk_id)
            key: Any = cluster_id if cluster_id is not None else chunk_id or result.get("document")
            if key in seen:
                continue
            seen.add(key)
            if cluster_id is not None:
                representative = self.dedup_representatives.get(cluster_id)
                if representative and chunk_id and representative["chunk_id"] != chunk_id:
                    result = self._promote_representative(result, representative, cluster_id)
            deduped.append(result)
        return deduped

    def _dedup_by_content(self, results: List[Dict]) -> List[Dict]:
        seen_content: Set[str] = set()
        unique_results: List[Dict] = []
        for result in results:
            content = result.get("document", result.get("content", ""))
            if content in seen_content:
                continue
            seen_content.add(content)
            unique_results.append(result)
        return unique_results

    def _extract_chunk_id(self, result: Dict) -> Optional[str]:
        metadata = result.get("metadata") or {}
        for key in ("chunk_id", "id", "document_id"):
            value = metadata.get(key)
            if value:
                return str(value)
        return None

    def _promote_representative(self, result: Dict, rep: Dict[str, Any], cluster_id: int) -> Dict:
        updated = result.copy()
        metadata = dict(result.get("metadata") or {})
        metadata["dedup_cluster_id"] = cluster_id
        metadata["canonical_chunk_id"] = rep.get("chunk_id")
        updated["metadata"] = metadata
        updated["canonical_chunk_id"] = rep.get("chunk_id")
        return updated

    def get_loaded_documents(self) -> List[str]:
        """Get list of currently loaded document filenames."""
        return list(self.current_documents)

    def clear_current_session(self):
        """Clear current session document tracking."""
        self.current_documents.clear()

        # Also clear auto-merging retriever if available
        if self.auto_merging_retriever and hasattr(self.auto_merging_retriever, "clear_documents"):
            try:
                self.auto_merging_retriever.clear_documents()
                logger.info("Cleared auto-merging retriever documents")
            except Exception as e:
                logger.warning(f"Failed to clear auto-merging retriever: {e}")

        logger.info("Cleared current session document tracking")

    def debug_collection_info(self) -> Dict:
        """Get debug information about the collection."""
        try:
            count = self.collection.count()
            all_metadata = self.collection.get(include=["metadatas"])

            # Count documents by filename
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

    def _save_cache(self):
        """Save query cache to disk (for testing)."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        # Convert tuple keys to strings for JSON serialization
        serializable_cache = {str(k): v for k, v in self.query_cache.items()}
        with open(self.cache_file, "w") as f:
            json.dump(serializable_cache, f)

    def _load_cache(self):
        """Load query cache from disk (for testing)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
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
        if existing_docs.get("ids"):
            logger.info(f"Document {filename} with same content already exists in database")
            # Still add to current session tracking
            self.current_documents.add(filename)
            return True
        return False

    def _prepare_chunk_data(
        self, chunks: List[dict], filename: str, file_hash: str, filepath: str
    ) -> dict:
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

    def _extract_sentence_window_chunk(
        self, chunk: dict, index: int, filename: str, file_hash: str, filepath: str
    ):
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
        self, chunk: str, index: int, filename: str, file_hash: str, filepath: str
    ):
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
            raise EmbeddingGenerationError(
                error_msg, {"filename": filename, "text_count": len(texts)}
            ) from e

    def _validate_model_availability(self) -> None:
        """
        Validate that the embedding model is available.

        Raises:
            ModelNotAvailableError: If the model is not available
        """
        if not self.model:
            raise ModelNotAvailableError("embedding_model")

    def _validate_embeddings_result(
        self, embeddings: List[List[float]], texts: List[str], filename: str
    ) -> None:
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
                {
                    "filename": filename,
                    "expected_count": len(texts),
                    "actual_count": len(embeddings) if embeddings else 0,
                },
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
        # If configured, prefer stable file-hash based chunk IDs
        prefer_hash = config.get("deep_chunk_id_use_file_hash", True)
        for i, metadata in enumerate(metadatas):
            base = None
            if prefer_hash and metadata.get("file_hash"):
                base = metadata.get("file_hash")
            else:
                base = filename

            if self.use_sentence_window and "sentence_index" in metadata:
                # Use sentence-based ID for sentence windows
                sentence_idx = metadata["sentence_index"]
                chunk_ids.append(f"{base}_s{sentence_idx}")
            else:
                # Use chunk index for character-based
                chunk_ids.append(f"{base}_chunk_{i}")
        return chunk_ids

    def _add_chunks_to_collection(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict],
        chunk_ids: List[str],
        filename: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Add chunks to the FAISS vector store.

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
            ids=chunk_ids,
            trace_id=trace_id,
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
        query_embeddings = self.inference_threading.generate_embeddings_threaded(
            [query], self.model
        )
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
        # Retrieve more candidates to allow BM25 reranking to work effectively
        return top_k * INITIAL_RETRIEVAL_MULTIPLIER if self.use_sentence_window else top_k

    def _query_collection_for_candidates(
        self,
        query_embedding: List[float],
        initial_top_k: int,
        query: str = "",
        trace_id: Optional[str] = None,
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
            if self.semantic_cache:
                cached = self.semantic_cache.lookup(query_embedding, n_results=initial_top_k)
                if cached:
                    logger.info("Semantic cache hit; skipping vector query")
                    return cached

            results = self._execute_collection_query(
                query_embedding, initial_top_k, trace_id=trace_id
            )
            processed = self._process_query_results(results, query)
            if self.semantic_cache and processed:
                self.semantic_cache.add(query, query_embedding, processed)
            return processed
        except Exception as e:
            error_msg = f"Failed to query document collection: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(
                error_msg,
                "QUERY_FAILED",
                {
                    "query_embedding_length": len(query_embedding),
                    "top_k": initial_top_k,
                    "current_docs_count": len(self.current_documents),
                },
            ) from e

    def _execute_collection_query(
        self, query_embedding: List[float], initial_top_k: int, trace_id: Optional[str] = None
    ):
        """Execute the vector store collection query."""
        # If no documents in current session, search ALL documents in database
        # Otherwise, only search current session documents
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": initial_top_k,
            "include": ["documents", "metadatas", "distances", "ids"],
        }

        if self.current_documents:
            query_params["where"] = {"filename": {"$in": list(self.current_documents)}}
            logger.debug(f"Searching in session documents: {self.current_documents}")
        else:
            logger.debug("No session filter - searching all documents in database")

        # Add trace_id when calling vector store query to record vector.query events
        if trace_id is not None:
            query_params["trace_id"] = trace_id
        return self.collection.query(**query_params)

    def _process_query_results(self, results, query: str = "") -> List[dict]:
        """Process raw query results into candidate format with optional keyword boosting."""
        candidates = []
        if results["documents"] and results["metadatas"] and results["distances"]:
            # Get IDs if available
            ids = results.get("ids", [[]])[0] if "ids" in results else []

            for i, (doc, metadata, distance) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ):
                base_similarity = 1 - distance  # Convert distance to similarity

                # Apply keyword boost with detailed breakdown
                score_breakdown = self._apply_keyword_boost_detailed(doc, query, base_similarity)

                # Update metadata with score breakdown for tracking
                updated_metadata = metadata.copy()
                updated_metadata["score_breakdown"] = score_breakdown

                # Get document ID if available
                doc_id = ids[i] if i < len(ids) else None

                candidates.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": updated_metadata,
                        "similarity": score_breakdown["final_score"],
                        "base_similarity": base_similarity,  # Keep original for debugging
                    }
                )
        return candidates

    def _update_bm25_statistics(self, texts: List[str], doc_ids: List[str]) -> None:
        """
        Update BM25 statistics when documents are added.

        Args:
            texts: List of document texts
            doc_ids: List of document IDs
        """
        docs = []
        for doc_id, text in zip(doc_ids, texts):
            docs.append({"doc_id": doc_id, "text": text})

        self.bm25.add_documents(docs)

        logger.debug(f"Updated BM25 stats: {len(doc_ids)} chunks added.")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing punctuation and lowercasing."""
        # Remove punctuation and split into words
        words = re.findall(r"\b\w+\b", text.lower())
        # Remove common stop words
        stop_words = {
            "tell",
            "me",
            "about",
            "the",
            "what",
            "is",
            "a",
            "an",
            "and",
            "or",
            "describe",
            "explain",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
            "was",
            "were",
            "are",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "of",
            "at",
            "by",
            "for",
            "with",
            "from",
            "to",
            "in",
            "on",
        }
        return [w for w in words if w not in stop_words and len(w) > 2]

    def save_bm25_stats(self, output_path: str) -> None:
        try:
            self.bm25.save_stats(output_path)
            logger.info(f"Saved BM25 stats to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 stats to {output_path}: {e}")

    def load_bm25_stats(self, input_path: str) -> None:
        self.bm25.load_stats(input_path)

    def _apply_keyword_boost(self, document: str, query: str, base_similarity: float) -> float:
        if not query or not document:
            return base_similarity
        query_terms = self._tokenize(query)
        if not query_terms:
            return base_similarity
        doc_id = hashlib.md5(document.encode(), usedforsecurity=False).hexdigest()[:8]
        bm25_score = self.bm25.compute_score(query_terms, doc_id, document)
        normalized_bm25 = min(bm25_score / BM25_NORMALIZATION_FACTOR, 1.0)
        if normalized_bm25 > MIN_BM25_THRESHOLD:
            boost_factor = KEYWORD_BOOST_FACTOR * normalized_bm25
            combined_score = base_similarity + boost_factor
            combined_score = min(combined_score, 1.0)
            logger.debug(
                f"BM25 BOOST: raw={bm25_score:.3f}, norm={normalized_bm25:.3f}, semantic={base_similarity:.3f}, boost={boost_factor:.3f}, combined={combined_score:.3f}"
            )
            return combined_score
        else:
            return base_similarity

    def _apply_keyword_boost_detailed(
        self, document: str, query: str, base_similarity: float
    ) -> Dict[str, float]:
        if not query or not document:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0,
            }
        query_terms = self._tokenize(query)
        if not query_terms:
            return {
                "final_score": base_similarity,
                "semantic_score": base_similarity,
                "bm25_score": 0.0,
                "semantic_contribution": base_similarity,
                "bm25_contribution": 0.0,
            }
        doc_id = hashlib.md5(document.encode(), usedforsecurity=False).hexdigest()[:8]
        bm25_score = self.bm25.compute_score(query_terms, doc_id, document)
        normalized_bm25 = min(bm25_score / BM25_NORMALIZATION_FACTOR, 1.0)
        semantic_weight = SEMANTIC_WEIGHT_DETAILED
        bm25_weight = BM25_WEIGHT_DETAILED
        semantic_contribution = semantic_weight * base_similarity
        bm25_contribution = bm25_weight * normalized_bm25
        final_score = semantic_contribution + bm25_contribution
        final_score = min(final_score, 1.0)
        logger.debug(
            f"DETAILED SCORE: semantic={base_similarity:.3f} (contrib={semantic_contribution:.3f}), bm25_raw={bm25_score:.3f}, bm25_norm={normalized_bm25:.3f} (contrib={bm25_contribution:.3f}), final={final_score:.3f}"
        )
        return {
            "final_score": final_score,
            "semantic_score": base_similarity,
            "bm25_score": bm25_score,
            "semantic_contribution": semantic_contribution,
            "bm25_contribution": bm25_contribution,
        }

    def _retrieve_by_bm25(self, query: str, top_k: int) -> List[dict]:
        try:
            all_docs = self.collection.get(
                include=["documents", "metadatas", "ids"],
                where=(
                    {"filename": {"$in": list(self.current_documents)}}
                    if self.current_documents
                    else None
                ),
            )
            if not all_docs["documents"]:
                return []

            # Prepare docs for BM25Searcher
            docs_for_search = []
            ids_list = all_docs.get("ids", [])
            for i, (doc, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                doc_id = (
                    ids_list[i]
                    if i < len(ids_list)
                    else hashlib.md5(doc.encode(), usedforsecurity=False).hexdigest()[:8]
                )
                docs_for_search.append({"doc_id": doc_id, "text": doc, "metadata": metadata})

            # Use BM25Searcher to search
            results = self.bm25.search(query, top_k=top_k, docs=docs_for_search)

            # Format results back to what retriever expects
            scored_docs = []
            for r in results:
                scored_docs.append(
                    {
                        "id": r["doc_id"],
                        "document": r["text"],
                        "metadata": r["metadata"],
                        "similarity": r["similarity"],
                        "base_similarity": 0.0,
                        "bm25_score": r["similarity"]
                        * 15.0,  # Reverse normalization if needed, but similarity is already normalized
                    }
                )

            logger.info(
                f"BM25 retrieval: scored {len(scored_docs)} docs, returning top {min(top_k, len(scored_docs))}"
            )
            return scored_docs
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []

    def _combine_semantic_and_bm25(
        self,
        semantic_candidates: List[dict],
        bm25_candidates: List[dict],
        top_k: int,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List[dict]:
        # Use the retrieval strategy to combine results
        return self.retrieval_strategy.combine_results(
            semantic_candidates, bm25_candidates, top_k, semantic_weight, bm25_weight
        )

    def _apply_sentence_window_postprocessing(
        self, candidates: List[dict], top_k: int, query: str, strategy: Optional[Dict] = None
    ) -> List[dict]:
        use_reranker = strategy.get("use_reranker") if strategy is not None else True
        # Use retrieval strategy for postprocessing
        return self.retrieval_strategy.apply_postprocessing(
            candidates,
            top_k,
            query,
            window_postprocessor=self.window_postprocessor if self.use_sentence_window else None,
            reranker=self.reranker if self.use_sentence_window else None,
            use_reranker=use_reranker,
        )

    def _apply_window_postprocessing(self, candidates: List[dict]) -> List[dict]:
        if self.use_sentence_window and self.window_postprocessor:
            return self.window_postprocessor.postprocess_results(candidates)
        return candidates

    def _apply_reranking_if_available(
        self, candidates: List[dict], top_k: int, query: str, use_reranker: bool = True
    ) -> List[dict]:
        if not use_reranker:
            return candidates
        if self.use_sentence_window and self.reranker and len(candidates) > top_k:
            reranked = self.reranker.rerank(query, candidates, max_results=len(candidates))
            if reranked:
                return reranked
        return candidates

    def _add_test_document(self, fake_path: str, doc: str) -> bool:
        def _add_test_document_operation():
            filename = self._get_filename_from_path(fake_path)
            if self._check_test_document_loaded(fake_path, filename):
                return False
            file_hash = self._generate_content_hash(doc)
            if self._check_test_document_duplicate(file_hash, filename):
                return False
            return self._prepare_and_add_test_document(
                doc, filename, file_hash, fake_path, trace_id=None
            )

        return self.service_manager.execute_sync(
            "document_processing", _add_test_document_operation
        )

    def _check_test_document_loaded(self, fake_path: str, filename: str) -> bool:
        if self.is_document_loaded(fake_path):
            logger.info(f"Document {filename} already loaded in current session")
            return True
        return False

    def _generate_content_hash(self, doc: str) -> str:
        import hashlib

        return hashlib.md5(doc.encode(), usedforsecurity=False).hexdigest()

    def _check_test_document_duplicate(self, file_hash: str, filename: str) -> bool:
        existing_docs = self.collection.get(where={"file_hash": file_hash})
        if existing_docs["ids"]:
            logger.info(f"Document {filename} with same content already exists in database")
            self.current_documents.add(filename)
            return True
        return False

    def _prepare_and_add_test_document(
        self,
        doc: str,
        filename: str,
        file_hash: str,
        fake_path: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        embeddings = self._generate_test_embeddings(doc, filename)
        chunk_ids, metadatas = self._create_test_chunk_metadata(doc, filename, file_hash, fake_path)
        self._add_test_chunks_to_collection(
            embeddings, [doc], metadatas, chunk_ids, filename, trace_id=trace_id
        )
        return True

    def _generate_test_embeddings(self, doc: str, filename: str) -> List[List[float]]:
        logger.info(f"Generating embeddings for {filename} (1 chunk)")
        return self.inference_threading.generate_embeddings_threaded([doc], self.model)

    def _create_test_chunk_metadata(self, doc: str, filename: str, file_hash: str, fake_path: str):
        chunk_ids = [f"{filename}_chunk_0"]
        metadatas = [
            {
                "filename": filename,
                "file_hash": file_hash,
                "filepath": fake_path,
                "chunk_index": 0,
                "total_chunks": 1,
            }
        ]
        return chunk_ids, metadatas

    def _add_test_chunks_to_collection(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        chunk_ids: List[str],
        filename: str,
        trace_id: Optional[str] = None,
    ) -> None:
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=chunk_ids,
            trace_id=trace_id,
        )
        self.current_documents.add(filename)
        logger.info(f"Successfully added {filename} with 1 chunk")

    def _choose_retrieval_method(self, query: str) -> str:
        is_complex = self._analyze_query_complexity(query)
        if (
            self.auto_merge_for_complex
            and is_complex
            and self.use_auto_merging
            and self.auto_merging_retriever
        ):
            return "auto_merging"
        else:
            return "sentence_window"

    def _execute_retrieval(
        self,
        method: str,
        query: str,
        top_k: int,
        strategy: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict]:
        if method == "auto_merging":
            logger.info("Using auto-merging retrieval for complex query")
            try:
                return self._retrieve_auto_merging_safe(
                    query, top_k, strategy=strategy, trace_id=trace_id
                )
            except Exception as e:
                logger.error(f"Auto-merging retrieval failed: {e}, falling back to sentence window")
                return self._retrieve_sentence_window(
                    query, top_k, strategy=strategy, trace_id=trace_id
                )
        else:
            logger.info("Using sentence window retrieval")
            return self._retrieve_sentence_window(query, top_k, strategy=strategy)

    def _retrieve_auto_merging_safe(
        self,
        query: str,
        top_k: int,
        strategy: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict]:
        if not self.auto_merging_retriever:
            raise Exception("Auto-merging retriever not available")
        # If auto_merging retriever supports trace_id, attempt to pass it; otherwise ignore
        try:
            results = self.auto_merging_retriever.retrieve(query, top_k=top_k, trace_id=trace_id)
        except TypeError:
            results = self.auto_merging_retriever.retrieve(query, top_k=top_k)
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "document": result.get("document", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": result.get("similarity", 1.0),
                }
            )
        logger.info(f"Retrieved {len(formatted_results)} chunks using auto-merging")
        return formatted_results

    def _validate_document_for_addition(self, filepath: str, filename: str) -> None:
        if self._check_document_exists(filepath, filename):
            raise DocumentAlreadyExistsError(filename, {"filepath": filepath})
        file_hash = self._get_file_hash(filepath)
        if self._check_database_duplicate(file_hash, filename):
            raise DocumentAlreadyExistsError(filename, {"file_hash": file_hash})

    def _process_and_add_document(
        self, chunk_data: dict, filename: str, trace_id: Optional[str] = None
    ) -> bool:
        embeddings = self._generate_chunk_embeddings(chunk_data["texts"], filename)
        chunk_ids = self._create_chunk_ids(chunk_data["metadatas"], filename)
        self._add_chunks_to_collection(
            embeddings,
            chunk_data["texts"],
            chunk_data["metadatas"],
            chunk_ids,
            filename,
            trace_id=trace_id,
        )
        return True


class FaissHybridRetriever:
    """
    Hybrid Retriever that combines sparse (BM25) and dense (FAISS) retrieval.

    The class provides an optional `reranker` parameter (e.g. CrossEncoder/LocalReranker) to
    perform re-ranking of the fused candidate list when a strategy requests it. The reranker
    must implement a `rerank(query, candidates, max_results=None)` method and may return a
    list of candidates that contain either `doc_id` (preferred) or `content`/`document` keys.

    When `strategy['use_reranker']` is True, the reranker will be invoked and its result
    used to produce the final ordered list of documents. If the reranker fails or is not
    available, the retriever falls back to the fused BM25/FAISS ordering.
    """

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        faiss_manager: FAISSIndexManager,
        embedding_generator: EmbeddingGenerator,
        documents: List[Dict],
        reranker=None,
    ):
        self.bm25_searcher = bm25_searcher
        self.faiss_manager = faiss_manager
        self.embedding_generator = embedding_generator
        self.documents = {doc["doc_id"]: doc for doc in documents}
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10, strategy: Optional[dict] = None) -> List[Dict]:
        """
        Performs hybrid search and returns a list of documents.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            strategy: Optional dict with weights and settings.

        Returns:
            List of document dicts with scores.
        """
        # Parse strategy parameters
        params = self._parse_strategy_params(strategy, top_k)

        # Get results from both retrievers
        bm25_results = self.bm25_searcher.search(query, top_k=params["bm25_k"])
        query_embedding = self.embedding_generator.encode([query])[0]
        faiss_results = self.faiss_manager.search(query_embedding, k=top_k)

        # Fuse results
        fused_results = self._fuse_retrieval_results(bm25_results, faiss_results, params, top_k)

        # Get full documents
        final_results = self._get_documents_from_results(fused_results, top_k)

        # Apply reranking if requested
        if params["use_reranker"] and self.reranker and final_results:
            return self._apply_reranking(query, final_results, top_k)

        return final_results

    def _parse_strategy_params(self, strategy: Optional[dict], top_k: int) -> Dict[str, Any]:
        """Parse strategy parameters with defaults.

        Args:
            strategy: Optional strategy dict.
            top_k: Default k value.

        Returns:
            Dict with parsed parameters.
        """
        if not strategy:
            return {
                "bm25_k": top_k,
                "faiss_k": top_k,
                "bm25_weight": 0.5,
                "dense_weight": 0.5,
                "use_reranker": False,
            }

        return {
            "bm25_k": strategy.get("k_candidates", top_k),
            "faiss_k": strategy.get("k_candidates", top_k),
            "bm25_weight": float(strategy.get("bm25_weight", 0.5)),
            "dense_weight": float(strategy.get("dense_weight", 0.5)),
            "use_reranker": bool(strategy.get("use_reranker", False)),
        }

    def _fuse_retrieval_results(
        self,
        bm25_results: List[Dict],
        faiss_results: List[Dict],
        params: Dict[str, Any],
        top_k: int,
    ) -> List[Dict]:
        """Fuse BM25 and FAISS results.

        Args:
            bm25_results: Results from BM25 searcher.
            faiss_results: Results from FAISS search.
            params: Strategy parameters with weights.
            top_k: Maximum results.

        Returns:
            Fused and sorted results.
        """
        try:
            from src.cubo.retrieval.fusion import combine_semantic_and_bm25

            fused = combine_semantic_and_bm25(
                faiss_results,
                bm25_results,
                semantic_weight=params["dense_weight"],
                bm25_weight=params["bm25_weight"],
                top_k=top_k,
            )
        except Exception:
            fused = self._fuse_results(bm25_results, faiss_results)

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    def _get_documents_from_results(self, fused_results: List[Dict], top_k: int) -> List[Dict]:
        """Get full documents from fused results.

        Args:
            fused_results: Fused search results with doc_ids.
            top_k: Maximum results.

        Returns:
            List of full document dicts with scores.
        """
        final_results = []
        for res in fused_results[:top_k]:
            doc_id = res["doc_id"]
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                doc["score"] = res["score"]
                final_results.append(doc)
        return final_results

    def _apply_reranking(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Apply reranking to candidate documents.

        Args:
            query: The search query.
            candidates: Candidate documents to rerank.
            top_k: Maximum results.

        Returns:
            Reranked documents or original candidates on failure.
        """
        try:
            normalized = self._normalize_candidates_for_reranking(candidates)
            reranked = self.reranker.rerank(query, normalized, max_results=len(normalized))
            if reranked:
                return self._map_reranked_to_documents(reranked, candidates, top_k)
        except Exception:
            pass
        return candidates

    def _normalize_candidates_for_reranking(self, candidates: List[Dict]) -> List[Dict]:
        """Normalize candidate documents for reranker input.

        Args:
            candidates: Raw candidate documents.

        Returns:
            Candidates with 'content' key set.
        """
        normalized = []
        for d in candidates:
            c = d.copy()
            if "content" not in c:
                c["content"] = c.get("text") or c.get("document", "")
            normalized.append(c)
        return normalized

    def _map_reranked_to_documents(
        self, reranked: List[Dict], original: List[Dict], top_k: int
    ) -> List[Dict]:
        """Map reranked results back to full documents.

        Args:
            reranked: Reranked candidate list.
            original: Original candidates for fallback matching.
            top_k: Maximum results.

        Returns:
            Documents with rerank scores.
        """
        output = []
        for c in reranked[:top_k]:
            doc = self._find_document_for_reranked(c, original)
            if doc:
                out = doc.copy()
                if "rerank_score" in c:
                    out["rerank_score"] = c["rerank_score"]
                output.append(out)
        return output

    def _find_document_for_reranked(
        self, reranked_item: Dict, original: List[Dict]
    ) -> Optional[Dict]:
        """Find the original document for a reranked item.

        Args:
            reranked_item: Reranked result item.
            original: Original candidates.

        Returns:
            Matching document or None.
        """
        # Try by doc_id first
        doc_id = reranked_item.get("doc_id")
        if doc_id and doc_id in self.documents:
            return self.documents[doc_id]

        # Fall back to content matching
        content = reranked_item.get("content") or reranked_item.get("document", "")
        for dd in original:
            dd_content = dd.get("text") or dd.get("content") or dd.get("document", "")
            if dd_content == content:
                return dd
        return None

    def _fuse_results(self, bm25_results: List[Dict], faiss_results: List[Dict]) -> List[Dict]:
        # Use the shared rrf_fuse util for standardization across retrievers
        return rrf_fuse(bm25_results, faiss_results)


# Backwards compatibility alias: importers can continue to use `HybridRetriever`.
HybridRetriever = FaissHybridRetriever

__all__ = [
    "DocumentRetriever",
    "FaissHybridRetriever",
    "HybridRetriever",
]
