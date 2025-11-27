import os
from pathlib import Path

import pytest

from src.cubo.config import config
from src.cubo.ingestion.fast_pass_ingestor import FastPassIngestor

try:

    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False


@pytest.mark.requires_whoosh
@pytest.mark.skipif(not WHOOSH_AVAILABLE, reason="Whoosh not installed")
def test_fastpass_whoosh_preserve_json(tmp_path: Path, monkeypatch):
    # Prepare input folder with a small text file
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    fpath = input_dir / "a.txt"
    fpath.write_text("apples and bananas")

    output_dir = tmp_path / "out"
    # Monkeypatch config using setattr on the _config dict
    monkeypatch.setattr(
        config,
        "_config",
        {
            **config._config,
            "bm25": {
                "backend": "whoosh",
                "whoosh_index_dir": "./whoosh_index",
                "preserve_bm25_stats_json": True,
            },
        },
    )
    # Run ingestion
    ingestor = FastPassIngestor(output_dir=str(output_dir))
    result = ingestor.ingest_folder(str(input_dir))
    assert "chunks_jsonl" in result
    assert "bm25_stats" in result
    # check that JSON exists
    assert os.path.exists(result["chunks_jsonl"])
    assert os.path.exists(result["bm25_stats"])
