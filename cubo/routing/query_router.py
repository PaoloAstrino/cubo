from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from cubo.config import config
from cubo.routing.llm_classifier import LLMQueryClassifier


class QueryType(Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievalStrategy:
    bm25_weight: float
    dense_weight: float
    use_reranker: bool
    k_candidates: int
    temporal_range: Optional[Tuple[Optional[datetime], Optional[datetime]]] = None
    classification: Optional[dict] = None


class QueryRouter:
    """Classifies queries and returns a retrieval strategy based on type and heuristics.

    This initial implementation uses regex patterns defined here; later iterations
    can plug in ML/LLM-based classifiers.
    """

    # Use a list to dictate precedence of patterns - more specific ones first
    _patterns = [
        (
            QueryType.TEMPORAL,
            re.compile(r"\b(when|date|year|in \d{4}|\d{4}-\d{4}|last|ago|recent|when did)\b", re.I),
        ),
        (
            QueryType.COMPARATIVE,
            re.compile(r"\b(compare|vs\b|versus|which is better|is .* better)\b", re.I),
        ),
        (
            QueryType.FACTUAL,
            re.compile(r"^(what is|who is|define|where is|how many|how much)\b", re.I),
        ),
        (QueryType.CONCEPTUAL, re.compile(r"\b(what does .* mean|why|how (does|do))\b", re.I)),
        (
            QueryType.EXPLORATORY,
            re.compile(r"\b(explain|tell me about|overview|summary|describe)\b", re.I),
        ),
    ]

    def __init__(self, presets: Optional[Dict[str, Any]] = None):
        self.presets = presets or config.get("query_router", {}).get("presets", {})
        # LLM fallback client (instantiable optionally)
        self.llm_enabled = bool(config.get("query_router", {}).get("use_llm_fallback", False))
        self.confidence_threshold = float(
            config.get("query_router", {}).get("confidence_threshold", 0.6)
        )
        self.llm_cache_ttl = int(config.get("query_router", {}).get("llm_cache_ttl_seconds", 300))
        self._llm_client = (
            LLMQueryClassifier(
                model_override=config.get("query_router", {}).get("llm_model"),
                timeout_ms=int(config.get("query_router", {}).get("llm_timeout_ms", 1000)),
            )
            if self.llm_enabled
            else None
        )
        # Simple in-memory TTL cache for LLM classification
        self._llm_cache = {}

    def classify(self, query: str) -> Tuple[QueryType, float, bool]:
        # Simple regex-based classifier with confidence heuristics
        text = (query or "").strip()
        if not text:
            return QueryType.EXPLORATORY, 0.5, False
        for qtype, pat in QueryRouter._patterns:
            if pat.search(text):
                # Confidence: higher if pattern is anchored or exact prefix
                confidence = 0.9 if pat.match(text) else 0.7
                # If confidence is low and LLM fallback enabled, call LLM classifier
                if confidence < self.confidence_threshold and self._llm_client:
                    # Check cache
                    cached = self._llm_cache.get(text)
                    now = time.time()
                    if cached and (now - cached["ts"] < self.llm_cache_ttl):
                        return cached["label"], cached["confidence"], True
                    try:
                        label_str, conf_llm, reason = self._llm_client.classify(text)
                        self._llm_cache[text] = {
                            "label": label_str,
                            "confidence": conf_llm,
                            "reason": reason,
                            "ts": now,
                        }
                        # Map to QueryType enum if possible
                        try:
                            qtype_llm = QueryType(label_str)
                        except Exception:
                            qtype_llm = QueryType.EXPLORATORY
                        # Prefer LLM if confidence higher
                        if conf_llm > confidence:
                            return qtype_llm, conf_llm, True
                    except Exception:
                        pass
                return qtype, confidence, False
        # Fallback: exploratory. If low confidence, call LLM fallback when enabled
        qtype = QueryType.EXPLORATORY
        confidence = 0.5
        if confidence < self.confidence_threshold and self._llm_client:
            cached = self._llm_cache.get(text)
            now = time.time()
            if cached and (now - cached["ts"] < self.llm_cache_ttl):
                return cached["label"], cached["confidence"], True
            try:
                label_str, conf_llm, reason = self._llm_client.classify(text)
                self._llm_cache[text] = {
                    "label": label_str,
                    "confidence": conf_llm,
                    "reason": reason,
                    "ts": now,
                }
                try:
                    qtype_llm = QueryType(label_str)
                except Exception:
                    qtype_llm = QueryType.EXPLORATORY
                return qtype_llm, conf_llm, True
            except Exception:
                pass
        return qtype, confidence, False

    def extract_temporal_filter(
        self, query: str
    ) -> Optional[Tuple[Optional[datetime], Optional[datetime]]]:
        # Very lightweight heuristics: capture explicit years or 'last N' constructs
        if not query:
            return None
        # Simple year range detection, e.g., 2019-2021
        m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", query)
        if m:
            try:
                start = datetime(int(m.group(1)), 1, 1)
                end = datetime(int(m.group(2)), 12, 31)
                return (start, end)
            except Exception:
                pass
        # Single year
        m = re.search(r"\b(\d{4})\b", query)
        if m:
            y = int(m.group(1))
            return (datetime(y, 1, 1), datetime(y, 12, 31))
        # 'last N days/months/years'
        m = re.search(r"last\s+(\d+)\s+(day|days|month|months|year|years)", query, re.I)
        if m:
            v = int(m.group(1))
            unit = m.group(2).lower()
            now = datetime.utcnow()
            if "day" in unit:
                start = now - timedelta(days=v)
            elif "month" in unit:
                start = now - timedelta(days=30 * v)
            else:
                start = now - timedelta(days=365 * v)
            return (start, now)
        return None

    def compute_strategy(self, query: str, context: Optional[dict] = None) -> RetrievalStrategy:
        qtype, conf, fallback_used = self.classify(query)
        preset = self.presets.get(qtype.value, {})
        bm25_weight = float(preset.get("bm25_weight", 0.5))
        dense_weight = float(preset.get("dense_weight", 0.5))
        use_reranker = bool(preset.get("use_reranker", False))
        k_candidates = int(preset.get("k_candidates", 100))
        temporal = None
        if qtype == QueryType.TEMPORAL:
            temporal = self.extract_temporal_filter(query)
        return RetrievalStrategy(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            use_reranker=use_reranker,
            k_candidates=k_candidates,
            temporal_range=temporal,
            classification={
                "label": qtype.value,
                "confidence": conf,
                "fallback_used": fallback_used,
            },
        )


query_router = QueryRouter()
