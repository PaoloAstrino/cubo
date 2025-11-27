#!/usr/bin/env python3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cubo.ingestion.fast_pass_ingestor import build_bm25_index


@pytest.fixture(scope="module")
def fast_pass_result():
    sample_folder = Path(__file__).parent.parent / "data"
    assert sample_folder.exists(), "Data folder not found for test"
    result = build_bm25_index(str(sample_folder), None, skip_model=True)
    assert result, "Fast pass ingest returned no result"
    assert result.get("chunks_jsonl"), "Chunks JSONL path missing"
    assert result.get("bm25_stats"), "BM25 stats not generated"
    return result


def test_fast_pass_ingest_sample_data(fast_pass_result):
    result = fast_pass_result
    assert "chunks_jsonl" in result and result["chunks_count"] > 0
    assert "ingestion_manifest.json" in str(result["bm25_stats"]).replace(
        "bm25_stats.json", "ingestion_manifest.json"
    )


def test_bm25_searcher_returns_results(fast_pass_result):
    from src.cubo.retrieval.bm25_searcher import BM25Searcher

    result = fast_pass_result
    searcher = BM25Searcher(result["chunks_jsonl"], result["bm25_stats"])
    res = searcher.search("whiskers", top_k=5)
    assert len(res) > 0
    assert result["bm25_stats"] is not None
