from pathlib import Path

import pytest

from src.cubo.retrieval.bm25_searcher import BM25Searcher


@pytest.mark.e2e
def test_ingest_and_retrieve(mini_data, fast_pass_result, mock_llm_client):
    """End-to-end smoke test: ingest mini_data, then perform BM25 search and a mocked generation."""
    # Validate ingestion result
    result = fast_pass_result
    assert result is not None
    chunks = result.get("chunks_jsonl")
    bm25_stats = result.get("bm25_stats")
    assert chunks is not None and Path(chunks).exists()
    assert bm25_stats is not None and Path(bm25_stats).exists()

    # Create BM25 searcher and run search queries
    searcher = BM25Searcher(chunks_jsonl=chunks, bm25_stats=bm25_stats)

    # Query for known terms
    res_paris = searcher.search("capital", top_k=3)
    assert len(res_paris) > 0
    # check that the text includes expected substring
    assert any("paris" in (r.get("text", "") or "").lower() for r in res_paris)

    res_whiskers = searcher.search("whiskers", top_k=3)
    assert len(res_whiskers) > 0
    assert any("whiskers" in (r.get("text", "") or "").lower() for r in res_whiskers)

    # Simulate generation with the mock generator
    answer = mock_llm_client.generate_response(
        "What is the capital of France?", res_paris[0].get("text", "")
    )
    assert isinstance(answer, str) and len(answer) > 0
