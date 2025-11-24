import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from src.cubo.config import config
from src.cubo.retrieval.retriever import DocumentRetriever


def _make_sample_df(tmpdir: Path):
    # Create a very small sample parquet used by the integration test
    df = pd.DataFrame(
        [
            {"chunk_id": "c1", "text": "alpha beta gamma", "summary_score": 0.1},
            {"chunk_id": "c2", "text": "alpha beta delta", "summary_score": 0.2},
            {"chunk_id": "c3", "text": "unique text one", "summary_score": 0.05},
            {"chunk_id": "c4", "text": "unique two here", "summary_score": 0.6},
        ]
    )
    path = tmpdir / "chunks.parquet"
    df.to_parquet(path)
    return path, df


def _make_embeddings(tmpdir: Path):
    # Deterministic embeddings such that c1/c2 are similar, others are different
    emb = np.array([
        [1.0, 0.5, 0.0],  # c1
        [1.0, 0.45, 0.0],  # c2
        [0.0, 0.0, 1.0],  # c3
        [0.0, 1.0, 0.5],  # c4
    ], dtype=np.float32)
    path = tmpdir / "embeddings.npy"
    np.save(path, emb)
    return path


def test_end_to_end_dedup_and_index_flow(tmp_path: Path, monkeypatch):
    tmpdir = tmp_path
    parquet_path, df = _make_sample_df(tmpdir)
    emb_path = _make_embeddings(tmpdir)

    # Output dedup map
    map_path = tmpdir / "dedup_map.json"

    # Run the deduplicate script programmatically via subprocess
    from subprocess import run, PIPE
    cmd = [
        str(Path(os.sys.executable)),
        "-m",
        "src.cubo.scripts.deduplicate",
        "--input-parquet",
        str(parquet_path),
        "--embeddings",
        str(emb_path),
        "--output-map",
        str(map_path),
        "--method",
        "hybrid",
        "--representative-metric",
        "summary_score",
        "--disable-prefilter",
    ]
    proc = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    assert proc.returncode == 0, f"deduplicate failed: {proc.stderr}"

    # Map file should exist and contain canonical_map / clusters / representatives
    with open(map_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    assert 'canonical_map' in payload
    assert 'clusters' in payload
    assert 'representatives' in payload

    # Confirm that c1 & c2 are clustered together (they were similar)
    clusters = payload['clusters']
    found_pair = any(set(['c1', 'c2']).issubset(set(m)) for m in clusters.values())
    assert found_pair, f"c1 and c2 should have been clustered together: clusters={clusters}"

    # Now validate _filter_to_representatives from build_faiss_index returns canonical rows only
    from src.cubo.scripts.build_faiss_index import _filter_to_representatives
    compact_df = _filter_to_representatives(pd.read_parquet(parquet_path), 'chunk_id', str(map_path))
    # All rows in compact_df should be canonical representatives
    rep_ids = {str(rep['chunk_id']) for rep in payload['representatives'].values()}
    assert set(compact_df['chunk_id'].astype(str).tolist()).issubset(rep_ids)

    # Set configuration for DocumentRetriever to pick up the map
    config.set('deduplication.map_path', str(map_path))
    config.set('deduplication.enabled', True)

    # Construct the retriever and ensure map has been loaded
    retriever = DocumentRetriever(model=None)
    assert retriever._dedup_map_loaded is True
    # Ensure that cluster lookup and canonical mapping have entries
    assert isinstance(retriever.dedup_cluster_lookup, dict)
    assert isinstance(retriever.dedup_canonical_lookup, dict)
    # Ensure canonical IDs exist in the lookup values
    assert set(retriever.dedup_canonical_lookup.values()).issubset(set(payload['canonical_map'].values()))


@pytest.mark.requires_faiss
def test_end_to_end_faiss_build_and_retrieval(tmp_path: Path):
    tmpdir = tmp_path
    parquet_path, df = _make_sample_df(tmpdir)
    emb_path = _make_embeddings(tmpdir)

    # Generate dedup map
    map_path = tmpdir / "dedup_map.json"
    from subprocess import run, PIPE
    cmd = [
        str(Path(os.sys.executable)),
        "-m",
        "src.cubo.scripts.deduplicate",
        "--input-parquet",
        str(parquet_path),
        "--embeddings",
        str(emb_path),
        "--output-map",
        str(map_path),
        "--method",
        "hybrid",
        "--representative-metric",
        "summary_score",
        "--disable-prefilter",
    ]
    proc = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    assert proc.returncode == 0, f"deduplicate failed: {proc.stderr}"

    # Filter to representative rows
    from src.cubo.scripts.build_faiss_index import _filter_to_representatives
    filtered_df = _filter_to_representatives(pd.read_parquet(parquet_path), 'chunk_id', str(map_path))

    # Map representative ids to embeddings
    emb = np.load(str(emb_path))
    id_to_idx = {row['chunk_id']: i for i, row in df.reset_index(drop=True).iterrows()}
    # Select embeddings in same order as filtered_df
    filtered_ids = list(map(str, filtered_df['chunk_id'].tolist()))
    filtered_embeddings = [emb[id_to_idx[i]].tolist() for i in filtered_ids]

    # Build FAISS index manager and save
    from src.cubo.indexing.faiss_index import FAISSIndexManager
    out_dir = tmpdir / "faiss_index"
    manager = FAISSIndexManager(dimension=emb.shape[1], index_dir=out_dir, nlist=4, hnsw_m=8, hot_fraction=0.5)
    manager.build_indexes(filtered_embeddings, filtered_ids)
    manager.save()

    # Reload and ensure metadata and ids are written
    manager2 = FAISSIndexManager(dimension=emb.shape[1], index_dir=out_dir)
    manager2.load(path=out_dir)
    # Combined IDs should match the number of representatives
    payload = json.loads(open(str(map_path), 'r', encoding='utf-8').read())
    expected_reps = set(str(rep.get('chunk_id')) for rep in payload.get('representatives', {}).values())
    saved_ids = set(manager2.hot_ids + manager2.cold_ids)
    assert saved_ids == expected_reps

    # Now perform a search using a query that matches c1's embedding and ensure results contain representative IDs
    query_vec = emb[0].tolist()
    results = manager2.search(query_vec, k=3)
    assert len(results) >= 1
    returned_ids = {r['id'] for r in results}
    assert saved_ids.intersection(returned_ids), f"Expected at least one representative to be returned; got {returned_ids}"


@pytest.mark.requires_faiss
def test_cli_build_faiss_and_retrieve(tmp_path: Path):
    tmpdir = tmp_path
    parquet_path, df = _make_sample_df(tmpdir)

    # Build a small FAISS index via CLI
    out_dir = tmpdir / 'faiss_cli_index'
    from subprocess import run, PIPE
    cmd = [
        str(Path(os.sys.executable)),
        '-m', 'src.cubo.scripts.build_faiss_index',
        '--parquet', str(parquet_path),
        '--index-dir', str(out_dir),
        '--batch-size', '2',
        '--nlist', '4',
        '--hnsw-m', '8',
    ]
    proc = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    assert proc.returncode == 0, f"build_faiss_index failed: {proc.stderr}"

    # Ensure metadata written
    meta = out_dir / 'metadata.json'
    assert meta.exists(), 'Expected metadata.json in index directory'
    payload = json.loads(open(str(meta), 'r', encoding='utf-8').read())
    ids_saved = set(payload.get('hot_ids', []) + payload.get('cold_ids', []))
    assert ids_saved, 'No ids saved into FAISS metadata'

    # Load and run a sample query to ensure index works
    from src.cubo.indexing.faiss_index import FAISSIndexManager
    manager = FAISSIndexManager(dimension=payload.get('dimension', 3), index_dir=out_dir)
    manager.load(path=out_dir)
    # Query with the embedding of the first chunk
    # Build deterministic embedding from text (same logic as script default embedder fallback uses lengths)
    import numpy as np
    query = np.zeros((payload.get('dimension', 3),), dtype=np.float32).tolist()
    res = manager.search(query, k=5)
    assert res, 'Expected at least one search result from FAISS index'
