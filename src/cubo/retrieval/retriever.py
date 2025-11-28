"""
CUBO Document Retriever - Facade over specialized retrieval components.

This module provides the main DocumentRetriever class which acts as a thin
facade over specialized components:
- DocumentStore: Document lifecycle management
- RetrievalExecutor: Core retrieval operations
- RetrievalOrchestrator: Tiered retrieval coordination
- RetrievalCacheService: Caching layer
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64

from sentence_transformers import SentenceTransformer

from src.cubo.config import config
from src.cubo.embeddings.model_inference_threading import get_model_inference_threading
from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.retrieval.cache import RetrievalCacheService
from src.cubo.retrieval.constants import DEFAULT_TOP_K, DEFAULT_WINDOW_SIZE
from src.cubo.retrieval.dependencies import (
    PostProcessorFactory,
    RerankerFactory,
    get_auto_merging_retriever,
    get_embedding_generator,
    get_scaffold_retriever,
    get_semantic_router,
    get_summary_embedder,
)
from src.cubo.retrieval.document_store import DocumentStore
from src.cubo.retrieval.orchestrator import (
    DeduplicationManager,
    HybridScorer,
    RetrievalOrchestrator,
    TieredRetrievalManager,
)
from src.cubo.retrieval.retrieval_executor import RetrievalExecutor, extract_chunk_id
from src.cubo.retrieval.strategy import RetrievalStrategy
from src.cubo.services.service_manager import get_service_manager
from src.cubo.storage.memory_store import InMemoryCollection
from src.cubo.utils.exceptions import CUBOError, DatabaseError, DocumentAlreadyExistsError, RetrievalError
from src.cubo.utils.logger import logger
from src.cubo.utils.trace_collector import trace_collector


class DocumentRetriever:
    """
    Facade for document retrieval using FAISS and sentence transformers.

    Delegates to specialized components:
    - DocumentStore: Document add/remove operations
    - RetrievalExecutor: Dense and sparse retrieval
    - RetrievalOrchestrator: Tiered retrieval coordination
    - RetrievalCacheService: Caching layer
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
        self.model = model
        self.service_manager = get_service_manager()
        self.inference_threading = get_model_inference_threading()
        self.use_sentence_window = use_sentence_window
        self.use_auto_merging = use_auto_merging
        self.auto_merge_for_complex = auto_merge_for_complex
        self.window_size = window_size
        self.top_k = top_k
        self._closed = False

        self._initialize_components()
        self._log_initialization_status()

    def _initialize_components(self) -> None:
        """Initialize all retrieval components."""
        self._setup_vector_store()
        self._setup_bm25()
        self._setup_caching()
        self._setup_document_store()
        self._setup_retrieval_executor()
        self._setup_deduplication()
        self._setup_auto_merging()
        self._setup_postprocessors()
        self._setup_tiered_retrieval()
        self._setup_router()

    def _setup_vector_store(self) -> None:
        """Setup FAISS vector store with fallback."""
        from src.cubo.retrieval.vector_store import create_vector_store

        backend = config.get("vector_store_backend", "faiss")
        model_dimension = (
            self.model.get_sentence_embedding_dimension()
            if self.model is not None
            else 384
        )

        try:
            self.collection = create_vector_store(
                backend=backend,
                dimension=model_dimension,
                index_dir=config.get("vector_store_path"),
                collection_name=config.get("collection_name", "cubo_documents"),
            )
        except Exception as e:
            logger.warning(f"Vector store init failed: {e}. Using in-memory fallback.")
            self.collection = InMemoryCollection()

    def _setup_bm25(self) -> None:
        """Setup BM25 searcher."""
        bm25_stats_path = config.get("bm25_stats_path", "data/bm25_stats.json")
        self.bm25 = BM25Searcher(bm25_stats=bm25_stats_path)
        if bm25_stats_path and os.path.exists(bm25_stats_path):
            logger.info(f"Loaded BM25 stats from {bm25_stats_path}")

    def _setup_caching(self) -> None:
        """Setup caching services."""
        cache_dir = config.get("cache_dir", "./cache")
        self.cache_service = RetrievalCacheService(
            cache_dir=cache_dir,
            semantic_cache_enabled=config.get("retrieval.semantic_cache.enabled", False),
            semantic_cache_ttl=int(config.get("retrieval.semantic_cache.ttl", 600)),
            semantic_cache_threshold=float(config.get("retrieval.semantic_cache.threshold", 0.93)),
            semantic_cache_max_entries=int(config.get("retrieval.semantic_cache.max_entries", 512)),
        )
        # Backwards compatibility
        self.query_cache = self.cache_service.query_cache
        self.cache_file = str(self.cache_service.query_cache_file)
        self.semantic_cache = self.cache_service.semantic_cache

    def _setup_document_store(self) -> None:
        """Setup document store."""
        self.document_store = DocumentStore(
            collection=self.collection,
            model=self.model,
            inference_threading=self.inference_threading,
            use_sentence_window=self.use_sentence_window,
            bm25_searcher=self.bm25,
        )

    def _setup_retrieval_executor(self) -> None:
        """Setup retrieval executor."""
        self.executor = RetrievalExecutor(
            collection=self.collection,
            bm25_searcher=self.bm25,
            model=self.model,
            inference_threading=self.inference_threading,
            semantic_cache=self.semantic_cache,
        )

    def _setup_deduplication(self) -> None:
        """Setup deduplication manager."""
        self.dedup_enabled = bool(config.get("deduplication.enabled", False))
        self.dedup_manager = DeduplicationManager.from_config(config)
        # Backwards compatibility
        self.dedup_cluster_lookup = self.dedup_manager.cluster_lookup
        self.dedup_representatives = self.dedup_manager.representatives
        self.dedup_canonical_lookup = self.dedup_manager.canonical_lookup
        self._dedup_map_loaded = self.dedup_manager._map_loaded

    def _setup_auto_merging(self) -> None:
        """Setup auto-merging retriever if enabled."""
        self.auto_merging_retriever = None
        if self.use_auto_merging:
            self.auto_merging_retriever = get_auto_merging_retriever(self.model)
            if not self.auto_merging_retriever:
                self.use_auto_merging = False

    def _setup_postprocessors(self) -> None:
        """Setup postprocessors and reranker."""
        self.window_postprocessor = None
        self.reranker = None
        if self.use_sentence_window:
            self.window_postprocessor = PostProcessorFactory.create_window_postprocessor()
            self.reranker = RerankerFactory.create_reranker(
                model=self.model,
                reranker_model_name=config.get("retrieval.reranker_model"),
                top_k=self.top_k,
            )
        self.retrieval_strategy = RetrievalStrategy()

    def _setup_tiered_retrieval(self) -> None:
        """Setup tiered retrieval components and orchestrator."""
        self.use_summary_prefilter = config.get("retrieval.use_summary_prefilter", False)
        self.use_scaffold_compression = config.get("retrieval.use_scaffold_compression", False)
        self.summary_prefilter_k = config.get("retrieval.summary_prefilter_k", 20)
        self.scaffold_weight = config.get("retrieval.scaffold_weight", 0.3)
        self.summary_weight = config.get("retrieval.summary_weight", 0.2)
        self.dense_weight = config.get("retrieval.dense_weight", 0.5)

        # Setup scaffold retriever
        self.scaffold_retriever = None
        if self.use_scaffold_compression:
            scaffold_dir = config.get("scaffold.output_dir", "./data/scaffolds")
            if Path(scaffold_dir).exists():
                embedding_gen = get_embedding_generator()
                if embedding_gen:
                    self.scaffold_retriever = get_scaffold_retriever(scaffold_dir, embedding_gen)

        # Setup summary embeddings
        self.summary_embeddings = None
        self.summary_chunk_ids = None
        if self.use_summary_prefilter:
            summary_dir = Path(config.get("summary_embeddings.output_dir", "./data/summary_embeddings"))
            if summary_dir.exists():
                embedder = get_summary_embedder()
                if embedder:
                    data = embedder.load_summary_embeddings(summary_dir)
                    self.summary_embeddings = data.get("embeddings")
                    self.summary_chunk_ids = data.get("chunk_ids", [])

        # Create orchestrator
        self.orchestrator = RetrievalOrchestrator(
            tiered_manager=TieredRetrievalManager(
                summary_embeddings=self.summary_embeddings,
                summary_chunk_ids=self.summary_chunk_ids or [],
                scaffold_retriever=self.scaffold_retriever,
                summary_weight=self.summary_weight,
                scaffold_weight=self.scaffold_weight,
                dense_weight=self.dense_weight,
            ),
            hybrid_scorer=HybridScorer(),
            dedup_manager=self.dedup_manager,
        )

    def _setup_router(self) -> None:
        """Setup query router."""
        self.router = get_semantic_router()

    def _log_initialization_status(self) -> None:
        """Log initialization status."""
        tiered = []
        if self.scaffold_retriever:
            tiered.append("scaffold")
        if self.summary_embeddings is not None:
            tiered.append("summary")
        tiered_str = f", tiered=[{','.join(tiered)}]" if tiered else ""
        logger.info(f"DocumentRetriever initialized (window={self.use_sentence_window}, auto_merge={self.use_auto_merging}{tiered_str})")

    # ========================================================================
    # Resource Management
    # ========================================================================

    def close(self) -> None:
        """Close and release resources."""
        try:
            self.cache_service.save_query_cache()
        except Exception:
            pass
        try:
            if hasattr(self.collection, "close"):
                self.collection.close()
            elif hasattr(self.collection, "reset"):
                self.collection.reset()
        except Exception:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ========================================================================
    # Document Operations (delegated to DocumentStore)
    # ========================================================================

    @property
    def current_documents(self) -> Set[str]:
        return self.document_store.current_documents

    @current_documents.setter
    def current_documents(self, value: Set[str]) -> None:
        self.document_store.current_documents = value

    @property
    def client(self):
        return getattr(self.collection, "client", None)

    def _get_file_hash(self, filepath: str) -> str:
        return self.document_store.get_file_hash(filepath)

    def _get_filename_from_path(self, filepath: str) -> str:
        return self.document_store.get_filename_from_path(filepath)

    def is_document_loaded(self, filepath: str) -> bool:
        return self.document_store.is_document_loaded(filepath)

    def get_loaded_documents(self) -> List[str]:
        return self.document_store.get_loaded_documents()

    def clear_current_session(self) -> None:
        self.document_store.clear_session()
        if self.auto_merging_retriever and hasattr(self.auto_merging_retriever, "clear_documents"):
            try:
                self.auto_merging_retriever.clear_documents()
            except Exception:
                pass

    def debug_collection_info(self) -> Dict:
        return self.document_store.debug_collection_info()

    def add_document(self, filepath: str, chunks: List[dict]) -> bool:
        """Add document chunks to the database."""
        try:
            return self.service_manager.execute_sync(
                "document_processing",
                lambda: self._do_add_document(filepath, chunks),
            )
        except DocumentAlreadyExistsError:
            return False
        except CUBOError:
            raise
        except Exception as e:
            raise DatabaseError(str(e), "ADD_DOCUMENT_FAILED", {"filepath": filepath}) from e

    def _do_add_document(self, filepath: str, chunks: List[dict]) -> bool:
        success = self.document_store.add_document(filepath, chunks)
        if success and self.auto_merging_retriever:
            try:
                self.auto_merging_retriever.add_document(filepath)
            except Exception:
                pass
        return success

    def add_documents(self, documents: list) -> bool:
        """Add multiple documents directly."""
        if not documents:
            return True
        added = False
        for i, doc in enumerate(documents):
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            path = doc.get("file_path", f"test_doc_{i}.txt") if isinstance(doc, dict) else f"test_doc_{i}.txt"
            if text and self._add_test_document(path, text):
                added = True
        return added

    def _add_test_document(self, fake_path: str, doc: str) -> bool:
        return self.service_manager.execute_sync(
            "document_processing",
            lambda: self.document_store.add_test_document(fake_path, doc),
        )

    def remove_document(self, filepath: str) -> bool:
        return self.document_store.remove_document(filepath)

    # ========================================================================
    # Retrieval Operations
    # ========================================================================

    def retrieve_top_documents(
        self,
        query: str,
        top_k: int = 6,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """Retrieve top-k relevant document chunks using hybrid retrieval."""
        try:
            if "k" in kwargs:
                try:
                    top_k = int(kwargs["k"])
                except Exception:
                    pass

            if trace_id:
                trace_collector.record(trace_id, "retriever", "start", {"query": query, "top_k": top_k})

            strategy = self.router.route_query(query) if self.router else None

            if self.use_auto_merging and self.auto_merging_retriever:
                results = self._hybrid_retrieval(query, top_k, strategy, trace_id)
            else:
                results = self._retrieve_sentence_window(query, top_k, strategy, trace_id)

            if trace_id:
                self._record_trace(trace_id, results)

            return results

        except CUBOError:
            raise
        except Exception as e:
            raise RetrievalError(str(e), "RETRIEVAL_FAILED", {"query": query[:100], "top_k": top_k}) from e

    def _hybrid_retrieval(self, query: str, top_k: int, strategy: Optional[Dict], trace_id: Optional[str]) -> List[Dict]:
        """Combine sentence window and auto-merging retrieval."""
        sentence_results = self._retrieve_sentence_window(query, top_k // 2 + top_k % 2, strategy, trace_id)
        auto_results = self._retrieve_auto_merging(query, top_k // 2)
        combined = sentence_results + auto_results
        unique = self.orchestrator.deduplicate_results(combined, extract_chunk_id)
        unique.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return unique[:top_k]

    def _retrieve_sentence_window(
        self,
        query: str,
        top_k: int,
        strategy: Optional[Dict] = None,
        trace_id: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve using sentence window with three-tier retrieval."""
        def _operation():
            if not self.document_store.has_documents():
                return []

            query_embedding = self.executor.generate_query_embedding(query)
            retrieval_k = int(strategy.get("k_candidates", top_k * 3)) if strategy else top_k * 3

            # Tiered retrieval
            summary_ids, scaffold_ids, scaffold_scores = self.orchestrator.execute_tiered_retrieval(
                query, np.array(query_embedding), self.summary_prefilter_k, max(5, top_k // 2)
            )

            # Dense + BM25 retrieval
            semantic = self.executor.query_dense(query_embedding, retrieval_k, query, self.current_documents, trace_id)
            bm25 = self.executor.query_bm25(query, retrieval_k, self.current_documents)

            # Combine
            bm25_weight = float(strategy.get("bm25_weight", 0.3)) if strategy else 0.3
            dense_weight = float(strategy.get("dense_weight", 0.7)) if strategy else 0.7
            combined = self.retrieval_strategy.combine_results(semantic, bm25, retrieval_k, dense_weight, bm25_weight)

            # Tiered boosting
            if summary_ids or scaffold_ids:
                combined = self.orchestrator.tiered_manager.apply_tiered_boosting(
                    combined, summary_ids, scaffold_ids, scaffold_scores, extract_chunk_id
                )

            # Postprocessing
            use_reranker = strategy.get("use_reranker", True) if strategy else True
            return self.retrieval_strategy.apply_postprocessing(
                combined, top_k, query,
                self.window_postprocessor if self.use_sentence_window else None,
                self.reranker if self.use_sentence_window else None,
                use_reranker,
            )

        return self.service_manager.execute_sync("database_operation", _operation)

    def _retrieve_auto_merging(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve using auto-merging."""
        if not self.auto_merging_retriever:
            return []
        try:
            results = self.auto_merging_retriever.retrieve(query, top_k=top_k)
            return [{"document": r.get("document", ""), "metadata": r.get("metadata", {}), "similarity": r.get("similarity", 1.0)} for r in results]
        except Exception:
            return []

    def _record_trace(self, trace_id: str, results: List[Dict]) -> None:
        try:
            trace_collector.record(trace_id, "retriever", "candidates", {
                "method": "hybrid" if self.use_auto_merging else "sentence_window",
                "candidates": [{"id": r.get("metadata", {}).get("id", ""), "similarity": r.get("similarity", 0)} for r in results[:10]],
            })
        except Exception:
            pass

    # ========================================================================
    # Backwards Compatibility
    # ========================================================================

    def _generate_query_embedding(self, query: str) -> List[float]:
        return self.executor.generate_query_embedding(query)

    def _query_collection_for_candidates(self, query_embedding: List[float], initial_top_k: int, query: str = "", trace_id: Optional[str] = None) -> List[dict]:
        return self.executor.query_dense(query_embedding, initial_top_k, query, self.current_documents, trace_id)

    def _retrieve_by_bm25(self, query: str, top_k: int) -> List[dict]:
        return self.executor.query_bm25(query, top_k, self.current_documents)

    def _apply_keyword_boost_detailed(self, document: str, query: str, base_similarity: float) -> Dict[str, float]:
        return self.executor.compute_hybrid_score(document, query, base_similarity)

    def _tokenize(self, text: str) -> List[str]:
        return self.executor._tokenize(text)

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        return self.orchestrator.deduplicate_results(results, extract_chunk_id)

    def _extract_chunk_id(self, result: Dict) -> Optional[str]:
        return extract_chunk_id(result)

    def _combine_semantic_and_bm25(self, semantic: List[dict], bm25: List[dict], top_k: int, semantic_weight: float = 0.7, bm25_weight: float = 0.3) -> List[dict]:
        return self.retrieval_strategy.combine_results(semantic, bm25, top_k, semantic_weight, bm25_weight)

    def _apply_reranking_if_available(self, candidates: List[dict], top_k: int, query: str, use_reranker: bool = True) -> List[dict]:
        if not use_reranker or not self.reranker or len(candidates) <= top_k:
            return candidates
        try:
            reranked = self.reranker.rerank(query, candidates, max_results=len(candidates))
            return reranked if reranked else candidates
        except Exception:
            return candidates

    def _create_chunk_ids(self, metadatas: List[dict], filename: str) -> List[str]:
        return self.document_store.create_chunk_ids(metadatas, filename)

    def _add_chunks_to_collection(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict],
        chunk_ids: List[str],
        filename: str,
    ) -> None:
        """Add chunks directly to the vector store (backwards compatibility)."""
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=chunk_ids,
        )
        self.current_documents.add(filename)
        self._update_bm25_statistics(texts, chunk_ids)
        logger.info(f"Successfully added {filename} with {len(chunk_ids)} chunks")

    def _update_bm25_statistics(self, texts: List[str], chunk_ids: List[str]) -> None:
        """Update BM25 statistics for keyword search."""
        if self.bm25:
            try:
                for text, chunk_id in zip(texts, chunk_ids):
                    self.bm25.add_document(chunk_id, text)
            except Exception:
                pass

    def _save_cache(self) -> None:
        self.cache_service.save_query_cache()

    def _load_cache(self) -> None:
        pass  # Loaded on init

    def save_bm25_stats(self, output_path: str) -> None:
        try:
            self.bm25.save_stats(output_path)
        except Exception as e:
            logger.error(f"Failed to save BM25 stats: {e}")

    def load_bm25_stats(self, input_path: str) -> None:
        self.bm25.load_stats(input_path)


# Keep FaissHybridRetriever for backwards compatibility
from src.cubo.retrieval.fusion import rrf_fuse
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator


class FaissHybridRetriever:
    """Hybrid Retriever combining BM25 and FAISS retrieval."""

    def __init__(self, bm25_searcher: BM25Searcher, faiss_manager: FAISSIndexManager, embedding_generator: EmbeddingGenerator, documents: List[Dict], reranker=None):
        self.bm25_searcher = bm25_searcher
        self.faiss_manager = faiss_manager
        self.embedding_generator = embedding_generator
        self.documents = {doc["doc_id"]: doc for doc in documents}
        self.reranker = reranker
        self.hybrid_scorer = HybridScorer()

    def search(self, query: str, top_k: int = 10, strategy: Optional[dict] = None) -> List[Dict]:
        bm25_results = self.bm25_searcher.search(query, top_k=top_k)
        query_embedding = self.embedding_generator.encode([query])[0]
        faiss_results = self.faiss_manager.search(query_embedding, k=top_k)
        fused = rrf_fuse(bm25_results, faiss_results)
        fused.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = [self.documents[r["doc_id"]] for r in fused[:top_k] if r.get("doc_id") in self.documents]
        
        # Apply reranking if enabled in strategy
        use_reranker = strategy.get("use_reranker", False) if strategy else False
        if use_reranker and self.reranker and results:
            try:
                reranked = self.reranker.rerank(query, results, max_results=len(results))
                if reranked:
                    results = reranked[:top_k]
            except Exception:
                pass
        
        return results


HybridRetriever = FaissHybridRetriever

__all__ = ["DocumentRetriever", "FaissHybridRetriever", "HybridRetriever"]
