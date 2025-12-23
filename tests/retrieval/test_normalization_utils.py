from cubo.retrieval.normalization import (
    compute_stats,
    minmax_normalize,
    normalize_scores,
    rank_normalize,
)


def test_minmax_normalize_basic():
    assert minmax_normalize([0.0, 1.0, 2.0]) == [0.0, 0.5, 1.0]


def test_minmax_normalize_constant_positive():
    assert minmax_normalize([2.0, 2.0]) == [1.0, 1.0]


def test_minmax_normalize_constant_zero():
    assert minmax_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_rank_normalize_shape():
    out = rank_normalize([10.0, 5.0, 0.0])
    assert out == [1.0, 0.5, 0.0]


def test_normalize_scores_none_passthrough():
    assert normalize_scores([1.2, -3.4], "none") == [1.2, -3.4]


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats.count == 0
    assert stats.min == 0.0
    assert stats.max == 0.0


def test_compute_stats_non_empty():
    stats = compute_stats([1.0, 3.0])
    assert stats.count == 2
    assert stats.min == 1.0
    assert stats.max == 3.0
