from cubo.routing.query_router import QueryRouter, QueryType


def test_classify_basic_cases():
    router = QueryRouter()
    cases = {
        "What is the capital of France?": QueryType.FACTUAL,
        "Compare apple vs banana in terms of vitamin content": QueryType.COMPARATIVE,
        "Give me an overview of quantum computing": QueryType.EXPLORATORY,
        "When did the Apollo 11 mission land?": QueryType.TEMPORAL,
        "How does CRISPR editing work?": QueryType.CONCEPTUAL,
    }
    for q, expected in cases.items():
        qtype, conf, fallback_used = router.classify(q)
        assert qtype == expected
        assert 0.0 <= conf <= 1.0
        assert isinstance(fallback_used, bool)
