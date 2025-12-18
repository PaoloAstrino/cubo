"""
Lazy dependency loading for retrieval components.

This module provides centralized lazy loading of optional dependencies
to avoid circular imports and improve startup time. Components that may
not always be available (like cross-encoder rerankers) are loaded on-demand.

Usage:
    from cubo.retrieval.dependencies import get_auto_merging_retriever

    retriever = get_auto_merging_retriever(model)
    if retriever is not None:
        results = retriever.retrieve(query)
"""

from typing import TYPE_CHECKING, Any, Optional

from cubo.utils.logger import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Cached module references
_auto_merging_class: Optional[type] = None
_cross_encoder_class: Optional[type] = None
_local_reranker_class: Optional[type] = None
_semantic_router_class: Optional[type] = None
_window_postprocessor_class: Optional[type] = None
_scaffold_retriever_factory: Optional[callable] = None
_summary_embedder_class: Optional[type] = None


def get_auto_merging_retriever(model: "SentenceTransformer") -> Optional[Any]:
    """
    Get an AutoMergingRetriever instance if available.

    Args:
        model: SentenceTransformer model for embeddings

    Returns:
        AutoMergingRetriever instance or None if not available
    """
    global _auto_merging_class

    if _auto_merging_class is None:
        try:
            from cubo.retrieval.custom_auto_merging import AutoMergingRetriever

            _auto_merging_class = AutoMergingRetriever
        except ImportError as e:
            logger.debug(f"AutoMergingRetriever not available: {e}")
            return None

    try:
        return _auto_merging_class(model)
    except Exception as e:
        logger.warning(f"Failed to initialize AutoMergingRetriever: {e}")
        return None


def get_cross_encoder_reranker(model_name: str, top_n: int = 10) -> Optional[Any]:
    """
    Get a CrossEncoderReranker instance if available.

    Args:
        model_name: Name of the cross-encoder model
        top_n: Number of results to return after reranking

    Returns:
        CrossEncoderReranker instance or None if not available
    """
    global _cross_encoder_class

    if _cross_encoder_class is None:
        try:
            from cubo.rerank.reranker import CrossEncoderReranker

            _cross_encoder_class = CrossEncoderReranker
        except ImportError as e:
            logger.debug(f"CrossEncoderReranker not available: {e}")
            return None

    try:
        return _cross_encoder_class(model_name=model_name, top_n=top_n)
    except Exception as e:
        logger.warning(f"Failed to initialize CrossEncoderReranker: {e}")
        return None


def get_local_reranker(model: "SentenceTransformer") -> Optional[Any]:
    """
    Get a LocalReranker instance.

    Args:
        model: SentenceTransformer model

    Returns:
        LocalReranker instance or None if not available
    """
    global _local_reranker_class

    if _local_reranker_class is None:
        try:
            from cubo.rerank.reranker import LocalReranker

            _local_reranker_class = LocalReranker
        except ImportError as e:
            logger.debug(f"LocalReranker not available: {e}")
            return None

    try:
        return _local_reranker_class(model)
    except Exception as e:
        logger.warning(f"Failed to initialize LocalReranker: {e}")
        return None


def get_semantic_router() -> Optional[Any]:
    """
    Get a SemanticRouter instance if available.

    Returns:
        SemanticRouter instance or None if not available
    """
    global _semantic_router_class

    if _semantic_router_class is None:
        try:
            from cubo.retrieval.router import SemanticRouter

            _semantic_router_class = SemanticRouter
        except ImportError as e:
            logger.debug(f"SemanticRouter not available: {e}")
            return None

    try:
        return _semantic_router_class()
    except Exception as e:
        logger.warning(f"Failed to initialize SemanticRouter: {e}")
        return None


def get_window_postprocessor() -> Optional[Any]:
    """
    Get a WindowReplacementPostProcessor instance if available.

    Returns:
        WindowReplacementPostProcessor instance or None if not available
    """
    global _window_postprocessor_class

    if _window_postprocessor_class is None:
        try:
            from cubo.processing.postprocessor import WindowReplacementPostProcessor

            _window_postprocessor_class = WindowReplacementPostProcessor
        except ImportError as e:
            logger.debug(f"WindowReplacementPostProcessor not available: {e}")
            return None

    try:
        return _window_postprocessor_class()
    except Exception as e:
        logger.warning(f"Failed to initialize WindowReplacementPostProcessor: {e}")
        return None


def get_scaffold_retriever(scaffold_dir: str, embedding_generator: Any) -> Optional[Any]:
    """
    Get a ScaffoldRetriever instance if available.

    Args:
        scaffold_dir: Directory containing scaffold data
        embedding_generator: EmbeddingGenerator instance

    Returns:
        ScaffoldRetriever instance or None if not available
    """
    global _scaffold_retriever_factory

    if _scaffold_retriever_factory is None:
        try:
            from cubo.retrieval.scaffold_retriever import create_scaffold_retriever_from_directory

            _scaffold_retriever_factory = create_scaffold_retriever_from_directory
        except ImportError as e:
            logger.debug(f"ScaffoldRetriever not available: {e}")
            return None

    try:
        retriever = _scaffold_retriever_factory(
            scaffold_dir=scaffold_dir,
            embedding_generator=embedding_generator,
        )
        if retriever and retriever.is_ready:
            return retriever
        logger.warning("Scaffold retriever loaded but not ready")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize ScaffoldRetriever: {e}")
        return None


def get_summary_embedder() -> Optional[Any]:
    """
    Get a SummaryEmbedder instance if available.

    Returns:
        SummaryEmbedder instance or None if not available
    """
    global _summary_embedder_class

    if _summary_embedder_class is None:
        try:
            from cubo.embeddings.summary_embedder import SummaryEmbedder

            _summary_embedder_class = SummaryEmbedder
        except ImportError as e:
            logger.debug(f"SummaryEmbedder not available: {e}")
            return None

    try:
        return _summary_embedder_class()
    except Exception as e:
        logger.warning(f"Failed to initialize SummaryEmbedder: {e}")
        return None


def get_embedding_generator() -> Optional[Any]:
    """
    Get an EmbeddingGenerator instance.

    Returns:
        EmbeddingGenerator instance or None if not available
    """
    try:
        from cubo.embeddings.embedding_generator import EmbeddingGenerator

        return EmbeddingGenerator()
    except ImportError as e:
        logger.debug(f"EmbeddingGenerator not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize EmbeddingGenerator: {e}")
        return None


class RerankerFactory:
    """Factory for creating reranker instances based on configuration."""

    @staticmethod
    def create_reranker(
        model: Optional["SentenceTransformer"] = None,
        reranker_model_name: Optional[str] = None,
        top_k: int = 10,
    ) -> Optional[Any]:
        """
        Create a reranker based on configuration.

        Priority:
        1. CrossEncoderReranker if model name is provided
        2. LocalReranker if model is provided
        3. None if neither is available

        Args:
            model: SentenceTransformer for LocalReranker
            reranker_model_name: Cross-encoder model name
            top_k: Number of results to return

        Returns:
            Reranker instance or None
        """
        # Try cross-encoder first
        if reranker_model_name and isinstance(reranker_model_name, str):
            reranker = get_cross_encoder_reranker(reranker_model_name, top_k)
            if reranker:
                logger.info(f"Using CrossEncoderReranker: {reranker_model_name}")
                return reranker
            logger.warning(
                "Failed to initialize CrossEncoderReranker, " "falling back to LocalReranker"
            )

        # Fall back to local reranker
        if model:
            reranker = get_local_reranker(model)
            if reranker:
                logger.info("Using LocalReranker")
                return reranker

        logger.warning("No reranker available")
        return None


class PostProcessorFactory:
    """Factory for creating post-processor instances."""

    @staticmethod
    def create_window_postprocessor(enabled: bool = True) -> Optional[Any]:
        """
        Create a window post-processor if enabled.

        Args:
            enabled: Whether to enable the post-processor

        Returns:
            WindowReplacementPostProcessor or None
        """
        if not enabled:
            return None
        return get_window_postprocessor()


# Convenience function to clear cached classes (useful for testing)
def clear_dependency_cache() -> None:
    """Clear all cached dependency classes."""
    global _auto_merging_class, _cross_encoder_class, _local_reranker_class
    global _semantic_router_class, _window_postprocessor_class
    global _scaffold_retriever_factory, _summary_embedder_class

    _auto_merging_class = None
    _cross_encoder_class = None
    _local_reranker_class = None
    _semantic_router_class = None
    _window_postprocessor_class = None
    _scaffold_retriever_factory = None
    _summary_embedder_class = None
