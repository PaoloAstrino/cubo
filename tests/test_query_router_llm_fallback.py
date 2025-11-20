from src.cubo.routing.query_router import QueryRouter, QueryType


class FakeLLMClient:
    def __init__(self, label=QueryType.FACTUAL, confidence=0.95):
        self.called = 0
        self.label = label
        self.confidence = confidence

    def classify(self, query: str):
        self.called += 1
        return self.label, self.confidence, 'mocked'


def test_llm_not_called_when_confidence_high(monkeypatch):
    router = QueryRouter()
    router.confidence_threshold = 0.6
    fake = FakeLLMClient()
    router._llm_client = fake
    # The factual query matches a pattern with high confidence (0.9) so LLM should not be called
    qtype, conf, fallback_used = router.classify('What is the capital of Spain?')
    assert qtype == QueryType.FACTUAL
    assert conf >= 0.9
    assert fallback_used is False
    assert fake.called == 0


def test_llm_called_on_low_conf(monkeypatch):
    router = QueryRouter()
    router.confidence_threshold = 0.6
    fake = FakeLLMClient(label=QueryType.CONCEPTUAL, confidence=0.92)
    router._llm_client = fake
    # This filler query does not match known patterns -> low regex confidence -> LLM called
    qtext = 'mysterious gibberish phrase for testing'
    qtype, conf, fallback_used = router.classify(qtext)
    assert fallback_used is True
    assert qtype == QueryType.CONCEPTUAL
    assert fake.called == 1
    # Call again to ensure caching avoids subsequent LLM call
    qtype2, conf2, fallback_used2 = router.classify(qtext)
    assert fake.called == 1
    assert fallback_used2 is True
