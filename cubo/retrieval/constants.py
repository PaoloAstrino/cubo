"""
Retrieval system constants (BRIDGE FILE).

This file now acts as a compatibility bridge, redirecting old constant
imports to the new dynamic Pydantic settings.
"""

from cubo.config.settings import settings

# =============================================================================
# BM25 PARAMETERS
# =============================================================================

BM25_K1 = settings.retrieval.bm25_k1
BM25_B = settings.retrieval.bm25_b
BM25_NORMALIZATION_FACTOR = settings.retrieval.bm25_normalization_factor
RRF_K = settings.retrieval.rrf_k

# =============================================================================
# RETRIEVAL PARAMETERS
# =============================================================================

DEFAULT_TOP_K = settings.retrieval.default_top_k
DEFAULT_WINDOW_SIZE = settings.retrieval.default_window_size
INITIAL_RETRIEVAL_MULTIPLIER = settings.retrieval.initial_retrieval_multiplier
MIN_CANDIDATE_POOL_SIZE = settings.retrieval.min_candidate_pool_size
COMPLEXITY_LENGTH_THRESHOLD = settings.retrieval.complexity_length_threshold

# =============================================================================
# FUSION WEIGHTS
# =============================================================================

SEMANTIC_WEIGHT_DEFAULT = settings.retrieval.semantic_weight_default
BM25_WEIGHT_DEFAULT = settings.retrieval.bm25_weight_default
SEMANTIC_WEIGHT_DETAILED = settings.retrieval.semantic_weight_detailed
BM25_WEIGHT_DETAILED = settings.retrieval.bm25_weight_detailed

# =============================================================================
# SCORING THRESHOLDS
# =============================================================================

MIN_BM25_THRESHOLD = settings.retrieval.min_bm25_threshold
KEYWORD_BOOST_FACTOR = settings.retrieval.keyword_boost_factor

# =============================================================================
# CACHE PARAMETERS
# =============================================================================

DEFAULT_CACHE_SIZE = settings.retrieval.cache_size
DEFAULT_CACHE_TTL = settings.retrieval.cache_ttl

# =============================================================================
# BATCH PROCESSING
# =============================================================================

# Keep these hardcoded for now or add to settings if needed
DEFAULT_BATCH_SIZE = 100
MAX_CONCURRENT_OPS = 4
