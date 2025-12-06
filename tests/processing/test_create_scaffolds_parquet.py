import pytest
pytest.importorskip("torch")

import json
from pathlib import Path

import pandas as pd

from cubo.processing.enrichment import ChunkEnricher
from cubo.processing.scaffold import create_scaffolds_from_parquet
from cubo.storage.metadata_manager import get_metadata_manager


class FakeLLM:
    """Fake LLM provider for testing"""

    def generate_response(self, prompt, context):
        return "Test summary"


def test_create_scaffolds_from_parquet_with_run(tmp_path: Path):
    # Prepare a simple parquet file
    df = pd.DataFrame(
        [
            {
                "chunk_id": f"c{i}",
                "text": f"Text of chunk {i}",
                "filename": f"doc{i}.txt",
                "file_hash": f"hash{i}",
                "token_count": 5,
            }
            for i in range(6)
        ]
    )
    parquet_path = tmp_path / "chunks.parquet"
    df.to_parquet(parquet_path)

    out_dir = tmp_path / "scaffold_out"
    run_id = "run_parquet_test"
    # Create enricher (now mandatory)
    enricher = ChunkEnricher(llm_provider=FakeLLM())
    # call wrapper
    res = create_scaffolds_from_parquet(
        str(parquet_path), str(out_dir), enricher=enricher, run_id=run_id
    )

    # We expect a manifest in data/manifests or under output's manifest location
    manifest_path = Path(out_dir).parent / "manifests" / f"{run_id}_scaffold_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == run_id
    # DB entry exists
    manager = get_metadata_manager()
    latest = manager.get_latest_scaffold_run()
    assert latest is not None
    assert latest["id"] == run_id
