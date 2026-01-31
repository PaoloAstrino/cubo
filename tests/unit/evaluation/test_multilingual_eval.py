from pathlib import Path

from tools.benchmarking.run_multilingual_eval import run_multilingual_eval


def test_skips_missing_queries(tmp_path):
    missing = tmp_path / "no_queries.jsonl"
    res = run_multilingual_eval("miracl-de", missing, Path("data/legal_de"))
    assert res.get("skipped") is True
    assert res.get("reason") == "missing queries file"


def test_skips_missing_index(tmp_path):
    qfile = tmp_path / "queries.jsonl"
    qfile.write_text('{"_id":"q1","text":"test"}\n', encoding="utf-8")
    res = run_multilingual_eval("miracl-de", qfile, Path("nonexistent_index_dir"))
    assert res.get("skipped") is True
    assert res.get("reason") == "missing index"
