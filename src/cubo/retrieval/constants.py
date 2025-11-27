"""
Retrieval system constants.

Centralizes all magic numbers and tunable parameters for the retrieval system.
This makes the codebase more maintainable and parameters easier to discover/tune.
"""

# =============================================================================
# BM25 PARAMETERS
# =============================================================================

# Okapi BM25 tuning parameters (standard values)
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Empirical maximum BM25 score for normalization to [0, 1] range.
# This value was determined through experimentation on typical document corpora.
BM25_NORMALIZATION_FACTOR = 15.0

# Reciprocal Rank Fusion constant (controls influence of rank position)
RRF_K = 60

# =============================================================================
# RETRIEVAL PARAMETERS
# =============================================================================

# Default number of documents to retrieve
DEFAULT_TOP_K = 3

# Sentence window size for context expansion
DEFAULT_WINDOW_SIZE = 3

# Multiplier for initial retrieval before reranking.
# When using reranking, we retrieve top_k * this value first, then rerank.
INITIAL_RETRIEVAL_MULTIPLIER = 5

# Query complexity threshold (character length).
# Queries shorter than this are considered "simple" for routing purposes.
COMPLEXITY_LENGTH_THRESHOLD = 12

# =============================================================================
# FUSION WEIGHTS
# =============================================================================

# Default weights for hybrid search (balanced fusion)
SEMANTIC_WEIGHT_DEFAULT = 0.7
BM25_WEIGHT_DEFAULT = 0.3

# Weights for detailed/precision queries (favors keyword matching)
SEMANTIC_WEIGHT_DETAILED = 0.1
BM25_WEIGHT_DETAILED = 0.9

# =============================================================================
# SCORING THRESHOLDS
# =============================================================================

# Minimum normalized BM25 score to consider for boosting
MIN_BM25_THRESHOLD = 0.05

# Factor for keyword boost contribution
KEYWORD_BOOST_FACTOR = 0.3

# =============================================================================
# CACHE PARAMETERS
# =============================================================================

# Default cache size (number of entries)
DEFAULT_CACHE_SIZE = 100

# Default TTL for cached results (seconds)
DEFAULT_CACHE_TTL = 3600

# =============================================================================
# BATCH PROCESSING
# =============================================================================

# Default batch size for document processing
DEFAULT_BATCH_SIZE = 100

# Maximum concurrent operations for parallel processing
MAX_CONCURRENT_OPS = 4
