from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NormalizationStats:
    count: int
    min: float
    max: float
    mean: float
    std: float


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def compute_stats(values: Sequence[float]) -> NormalizationStats:
    if not values:
        return NormalizationStats(count=0, min=0.0, max=0.0, mean=0.0, std=0.0)
    mean, std = _mean_std(values)
    return NormalizationStats(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=mean,
        std=std,
    )


def minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        # No variance: keep deterministic. If scores are all >0, treat as strong.
        return [1.0 if vmax > 0 else 0.0 for _ in values]
    scale = vmax - vmin
    return [(v - vmin) / scale for v in values]


def rank_normalize(values: Sequence[float]) -> List[float]:
    """Normalize by rank only (top=1.0, bottom=0.0)."""
    if not values:
        return []
    if len(values) == 1:
        return [1.0]

    # Stable ranking: higher value => better rank.
    indexed = list(enumerate(values))
    indexed.sort(key=lambda t: t[1], reverse=True)

    ranks = [0] * len(values)
    for rank, (idx, _val) in enumerate(indexed, start=1):
        ranks[idx] = rank

    denom = len(values) - 1
    return [1.0 - ((r - 1) / denom) for r in ranks]


def zscore_normalize(values: Sequence[float]) -> List[float]:
    """Return z-scores (not bounded to [0,1]). Caller may map further."""
    if not values:
        return []
    mean, std = _mean_std(values)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def normalize_scores(values: Sequence[float], method: str) -> List[float]:
    method = (method or "minmax").strip().lower()
    if method == "none":
        return list(values)
    if method == "rank":
        return rank_normalize(values)
    if method == "zscore":
        # Map z-scores into [0,1] via a cheap logistic approximation.
        zs = zscore_normalize(values)
        # sigmoid(z) = 1/(1+exp(-z))
        import math

        return [1.0 / (1.0 + math.exp(-z)) for z in zs]
    # default
    return minmax_normalize(values)


def normalize_candidates(
    candidates: List[Dict[str, Any]],
    *,
    method: str,
    source: str,
    score_key: str = "similarity",
) -> Tuple[List[Dict[str, Any]], NormalizationStats]:
    """Annotate candidates with raw/normalized scores and return stats.

    Adds:
      - source
      - raw_similarity
      - normalized_similarity

    Also sets candidate[score_key] to normalized value for backward compatibility.
    """
    if not candidates:
        return candidates, compute_stats([])

    raw_scores: List[float] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        c.setdefault("source", source)
        if "raw_similarity" not in c:
            c["raw_similarity"] = c.get(score_key)
        raw_scores.append(_to_float(c.get("raw_similarity"), default=0.0))

    stats = compute_stats(raw_scores)
    normalized = normalize_scores(raw_scores, method)

    norm_iter = iter(normalized)
    for c in candidates:
        if not isinstance(c, dict):
            continue
        try:
            n = float(next(norm_iter))
        except StopIteration:
            n = 0.0
        c["normalized_similarity"] = n
        c[score_key] = n

    return candidates, stats
