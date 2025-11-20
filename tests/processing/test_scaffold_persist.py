from pathlib import Path
import pytest
import pandas as pd
import json

from src.cubo.processing.scaffold import ScaffoldGenerator, save_scaffold_run
from src.cubo.config import config
from src.cubo.storage.metadata_manager import get_metadata_manager


class FakeEmbeddingGenerator:
    def __init__(self, dim=8):
        self.dim = dim

    def encode(self, texts, batch_size=32):
        # Return deterministic embeddings for testing
        embeddings = []
        for i, _ in enumerate(texts):
            embeddings.append([float(i + j) for j in range(self.dim)])
        return embeddings


def test_save_scaffold_run_and_db(tmp_path: Path):
    # Setup temporary metadata DB
    config.set('metadata_db_path', str(tmp_path / 'metadata.db'))
    # Create simple chunks DataFrame
    df = pd.DataFrame([
        {'chunk_id': f'c{i}', 'text': f'Text of chunk {i}', 'filename': f'doc{i}.txt', 'file_hash': f'hash{i}', 'token_count': 3 + i} for i in range(8)
    ])

    generator = ScaffoldGenerator(enricher=None, embedding_generator=FakeEmbeddingGenerator(dim=4), scaffold_size=3)
    scaffolds_result = generator.generate_scaffolds(df, text_column='text', id_column='chunk_id')

    # Save scaffold run
    output_root = tmp_path / 'scaffolds'
    run_id = 'scaffold_test_1'
    res = save_scaffold_run(run_id, scaffolds_result, output_root=output_root, model_version='v1', input_chunks_df=df, id_column='chunk_id')

    run_dir = Path(res['run_dir'])
    manifest_path = Path(res['manifest'])

    # Files exist
    assert (run_dir / 'scaffold_metadata.parquet').exists()
    assert (run_dir / 'scaffold_mapping.json').exists()
    assert manifest_path.exists()

    # DB entries
    manager = get_metadata_manager()
    latest = manager.get_latest_scaffold_run()
    assert latest is not None
    assert latest['id'] == run_id
    mappings = manager.list_scaffold_mappings_for_run(run_id)
    # There should be mappings for each scaffold chunk pair
    assert len(mappings) >= 1
    # Validate parquet contains compression and model_version
    sc_df = pd.read_parquet(run_dir / 'scaffold_metadata.parquet')
    for k in ['compression_ratio', 'original_size', 'compressed_size', 'original_token_count', 'compressed_token_count', 'model_version']:
        assert k in sc_df.columns
    # Validate manifest includes file metadata for at least one chunk
    manifest = json.loads(manifest_path.read_text())
    assert 'chunks_summary' in manifest
    assert len(manifest['chunks_summary']) >= 1
    assert 'file_hash' in manifest['chunks_summary'][0]
    assert 'filename' in manifest['chunks_summary'][0]
    assert 'token_count' in manifest['chunks_summary'][0]
