import pytest
from src.cubo.retrieval.router import SemanticRouter, QueryType
from src.cubo.config import config


def test_router_classify_basic_queries():
    router = SemanticRouter()
    assert router.classify_query("What is the capital of France?") == QueryType.FACTUAL
    assert router.classify_query("Explain how neural networks work.") == QueryType.CONCEPTUAL
    assert router.classify_query("Compare X vs Y") == QueryType.COMPARATIVE
    assert router.classify_query("Latest developments 2023") == QueryType.TEMPORAL
    assert router.classify_query("") == QueryType.EXPLORATORY


def test_router_route_query_defaults():
    router = SemanticRouter()
    strategy = router.route_query("What is the capital of France?")
    assert strategy['query_type'] == 'factual'
    assert 'bm25_weight' in strategy and 'dense_weight' in strategy
    assert strategy['bm25_weight'] >= 0 and strategy['dense_weight'] >= 0
