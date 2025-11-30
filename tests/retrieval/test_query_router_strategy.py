from cubo.config import config
from cubo.routing.query_router import QueryRouter, RetrievalStrategy


def test_compute_strategy_matches_config():
    # Ensure strategy presets map to expected default config values
    router = QueryRouter()
    presets = config.get("query_router", {}).get("presets", {})
    for qtype_name, preset in presets.items():
        # pick a canonical query for qtype
        if qtype_name == "factual":
            query = "What is the capital of Spain?"
        elif qtype_name == "comparative":
            query = "Which is better: A or B?"
        elif qtype_name == "exploratory":
            query = "Tell me about the causes of climate change"
        elif qtype_name == "temporal":
            query = "What happened in 1945?"
        else:
            query = "Explain Newton's laws"
        strategy = router.compute_strategy(query)
        assert isinstance(strategy, RetrievalStrategy)
        assert strategy.bm25_weight == float(preset.get("bm25_weight"))
        assert strategy.dense_weight == float(preset.get("dense_weight"))
        assert strategy.use_reranker == bool(preset.get("use_reranker"))
        # classification metadata
        assert strategy.classification is not None
        assert (
            "label" in strategy.classification
            and "confidence" in strategy.classification
            and "fallback_used" in strategy.classification
        )
