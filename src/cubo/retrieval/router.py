"""Semantic Router responsible for classifying the query and returning a retrieval strategy.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Dict, Any

from src.cubo.config import config


class QueryType(Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    EXPLORATORY = "exploratory"


class SemanticRouter:
    """Simple semantic router using regex patterns and light heuristics.

    This class is configurable via the `routing` section of the config.
    It produces a strategy dictionary with weights and flags used by the retriever.
    """

    def __init__(self):
        # Load config defaults
        self.config = config
        self.patterns = self._build_default_patterns()

    def _build_default_patterns(self):
        return {
            QueryType.FACTUAL: [
                r"\bwhat is\b", r"\bwho is\b", r"\bwhen did\b",
                r"\bhow many\b", r"\bdefine\b", r"\blist\b"
            ],
            QueryType.TEMPORAL: [
                r"\brecent\b", r"\blatest\b", r"\blast\b",
                r"\b20\d{2}\b", r"\bthis year\b"
            ],
            QueryType.COMPARATIVE: [
                r"\bcompare\b", r"\bversus\b", r"\bvs\b",
                r"\bdifference between\b", r"\bbetter than\b"
            ],
            QueryType.CONCEPTUAL: [
                r"\bexplain\b", r"\bwhy\b", r"\bhow does\b",
                r"\brelationship\b", r"\bimpact\b"
            ],
        }

    def classify_query(self, query_text: str) -> QueryType:
        """Return the detected QueryType based on pattern matches."""
        if not query_text or not isinstance(query_text, str):
            return QueryType.EXPLORATORY

        q_low = query_text.lower()
        for q_type, patterns in self.patterns.items():
            for p in patterns:
                if re.search(p, q_low):
                    return q_type

        return QueryType.EXPLORATORY

    def extract_temporal_filter(self, query_text: str):
        try:
            import dateparser
        except Exception:
            dateparser = None
        # Very small attempt at temporal extraction; use config or a full NL parser for better results
        if not query_text:
            return None
        match = re.search(r"\b(20\d{2})\b", query_text)
        if match and dateparser:
            try:
                return dateparser.parse(match.group(1))
            except Exception:
                return None
        # Check for relative terms
        if re.search(r"\b(last|this|past) (week|month|year)\b", query_text.lower()):
            return "relative"
        return None

    def route_query(self, query_text: str) -> Dict[str, Any]:
        """Return a retrieval strategy according to the query type and config defaults.

        Strategy keys:
          - query_type
          - temporal_filter
          - use_bm25
          - bm25_weight
          - dense_weight
          - use_reranker
          - k_candidates
        """
        q_type = self.classify_query(query_text)
        temporal = self.extract_temporal_filter(query_text)

        # Base defaults. Allow overriding via config
        defaults = {
            'bm25_weight': float(self.config.get('routing.factual_bm25_weight', 0.6)),
            'dense_weight': float(self.config.get('routing.conceptual_dense_weight', 0.8)),
            'k_candidates': int(self.config.get('retrieval.bm25_candidates', 500)),
            'use_reranker': bool(self.config.get('retrieval.use_reranker', False)),
        }

        # Adjust defaults by query type
        strategy = {
            'query_type': q_type.value,
            'temporal_filter': temporal,
            'use_bm25': True,
            'bm25_weight': 0.3,
            'dense_weight': 0.7,
            'use_reranker': defaults['use_reranker'],
            'k_candidates': defaults['k_candidates']
        }

        if q_type == QueryType.FACTUAL:
            strategy['bm25_weight'] = 0.6
            strategy['dense_weight'] = 0.4
            strategy['k_candidates'] = max(50, int(defaults['k_candidates'] * 0.6))
        elif q_type == QueryType.CONCEPTUAL:
            strategy['bm25_weight'] = 0.2
            strategy['dense_weight'] = 0.8
            strategy['use_reranker'] = True
            strategy['k_candidates'] = int(defaults['k_candidates'] * 0.8)
        elif q_type == QueryType.COMPARATIVE:
            strategy['use_reranker'] = True
            strategy['k_candidates'] = int(defaults['k_candidates'] * 0.5)
        elif q_type == QueryType.TEMPORAL:
            strategy['bm25_weight'] = 0.4
            strategy['dense_weight'] = 0.6
            strategy['k_candidates'] = int(defaults['k_candidates'] * 0.75)

        # Allow final override via config flags
        if self.config.get('routing.enable', True) is False:
            # Router disabled: use default retrieval parameters
            return {
                'query_type': QueryType.EXPLORATORY.value,
                'temporal_filter': None,
                'use_bm25': True,
                'bm25_weight': 0.3,
                'dense_weight': 0.7,
                'use_reranker': defaults['use_reranker'],
                'k_candidates': defaults['k_candidates']
            }

        return strategy
