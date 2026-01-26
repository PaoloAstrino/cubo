import logging

from cubo.ingestion.hierarchical_chunker import HierarchicalChunker
from cubo.monitoring import metrics
from cubo.utils.utils import Utils


def setup_function():
    metrics.reset_metrics()


def test_metrics_record(monkeypatch):
    # Simple token count stub: words = tokens
    monkeypatch.setattr(
        Utils, "_token_count", staticmethod(lambda text, tokenizer=None: len(text.split()))
    )
    metrics.reset_metrics()

    chunker = HierarchicalChunker(max_chunk_size=1000)
    chunker.chunk("One sentence.")

    m = metrics.get_metrics()
    assert m.get("chunks_created", 0) >= 1


def test_warning_on_token_threshold(caplog, monkeypatch):
    caplog.set_level(logging.WARNING)

    # Force token counts to hit the threshold
    monkeypatch.setattr(Utils, "_token_count", staticmethod(lambda text, tokenizer=None: 10))
    metrics.reset_metrics()

    chunker = HierarchicalChunker(max_chunk_tokens=10)
    chunker.chunk("Sentence A. Sentence B.")

    assert any("approaching token limit" in rec.getMessage() for rec in caplog.records)
