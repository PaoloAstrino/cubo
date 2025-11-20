from pathlib import Path
import json
import pandas as pd

from src.cubo.processing.scaffold import create_scaffolds_from_parquet
from src.cubo.config import config
from src.cubo.storage.metadata_manager import get_metadata_manager


def test_create_scaffolds_from_parquet_with_run(tmp_path: Path):
    # Prepare a simple parquet file
    df = pd.DataFrame([
        {'chunk_id': f'c{i}', 'text': f'Text of chunk {i}', 'filename': f'doc{i}.txt','file_hash': f'hash{i}', 'token_count': 5} for i in range(6)
    ])
    parquet_path = tmp_path / 'chunks.parquet'
    df.to_parquet(parquet_path)

    out_dir = tmp_path / 'scaffold_out'
    run_id = 'run_parquet_test'
    # call wrapper
    res = create_scaffolds_from_parquet(str(parquet_path), str(out_dir), run_id=run_id)

    # We expect a manifest in data/manifests or under output's manifest location
    manifest_path = Path(out_dir).parent / 'manifests' / f"{run_id}_scaffold_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['run_id'] == run_id
    # DB entry exists
    manager = get_metadata_manager()
    latest = manager.get_latest_scaffold_run()
    assert latest is not None
    assert latest['id'] == run_id
