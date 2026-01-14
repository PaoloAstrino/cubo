# Deduplication (Semantic + Hybrid)

> **Category:** Feature | **Status:** Active

Explanation of the semantic and hybrid deduplication pipeline used to reduce index redundancy.

---

This document explains how the deduplication (semantic + hybrid) pipeline works in CUBO and how to use the CLI to generate dedup maps and apply them to FAISS index builds.

## Overview

- The deduplication pipeline combines a MinHash prefilter to reduce false positive edges and a nearest-neighbor approach (FAISS / HNSW / scikit-learn) to build a similarity graph.
- Clustering is performed using HDBSCAN (recommended) or as a connected component analysis
- Each cluster gets a representative document chosen by the configured `representative_metric`. By default, `summary_score` is preferred.
- The canonical map is written as a JSON file that contains `canonical_map`, `clusters`, and a top-level `representatives` map.

## Configuration

The main deduplication options live in `src/cubo/config.py` -> `deduplication`. Key options:

- `enabled` (bool) - enable dedup application in the `DocumentRetriever` at startup
- `method` (string) - `minhash`, `semantic`, or `hybrid`
- `run_on` (string) - `scaffold` or `chunks` depending on run granularity
- `representative_metric` (string) - `summary_score` or `text_length` etc.
- `similarity_threshold` (float) - cosine similarity threshold to keep ANN edges
- `map_path` (string) - path where `output/dedup_clusters.json` is stored
- `prefilter` - minhash options
- `ann` - backend and `k` (neighbors to query)
- `clustering` - `hdbscan` parameters and `umap_dims`

## Generating a Dedup Map

1. Make sure you have embeddings for the content you dedup (per-chunk or scaffold). The dedup CLI accepts a path to a `numpy` (`.npy`) file or a list-like object. Example:

```pwsh
python -m src.cubo.scripts.deduplicate \
  --input-parquet data/chunks.parquet \
  --embeddings data/chunk_embeddings.npy \
  --output-map output/dedup_clusters.json \
  --method hybrid \
  --representative-metric summary_score \
  --threshold 0.8
```

2. For a quick prefilter-only (MinHash) option run:

```pwsh
python -m src.cubo.scripts.deduplicate --input-parquet data/chunks.parquet --output-map out/minhash_map.json --method minhash
```

3. The CLI returns a JSON map, including clusters and the chosen representative for each cluster.

## Using the Dedup Map during Indexing

When building the FAISS index, you can pass the `--dedup-map` argument to filter the source chunks down to the canonical entries only.

```pwsh
python -m src.cubo.scripts.build_faiss_index --parquet data/chunks.parquet --dedup-map output/dedup_clusters.json --index-dir faiss_index --dry-run
```

This approach helps keep the index smaller and avoids duplicate context during retrieval.

## How the Retriever Uses the Map

If `deduplication.enabled` is True and `config.deduplication.map_path` is present, `DocumentRetriever` will load the map at initialization and mark documents with `cluster_id` and `is_representative` in the chunk DataFrame. Behavior:

- `dedup_canonical_lookup`: maps chunk_id -> canonical chunk id
- `dedup_cluster_lookup`: maps chunk_id -> cluster_id
- `dedup_representatives`: cluster_id -> representative metadata

Results returned to the caller may feature an additional `canonical_chunk_id` and `dedup_cluster_id` fields depending on the retrieval code path.

## Representative Metrics

Default is `summary_score` which uses the `summary_score` numeric field if present; otherwise the fallback is `text` length. You can configure this via `config.json` or pass `--representative-metric` to the CLI.

## Notes

- Optional dependencies: `faiss`, `hdbscan`, `umap-learn`, `hnswlib` — install as needed to use the desired backend and clustering approach.
- If you plan to run dedup pipelines in CI or on Windows, include `datasketch` in your environment as MinHash is used in prefiltering.

## Troubleshooting
## How to generate embeddings for the dedup CLI

You can use `EmbeddingGenerator` to produce deterministic, batched embeddings and save them as a `.npy` file. Example:

```python
from pathlib import Path
import numpy as np
import pandas as pd
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator

df = pd.read_parquet('data/chunks.parquet')
texts = df['text'].fillna('').astype(str).tolist()
generator = EmbeddingGenerator(batch_size=32)
embeddings = generator.encode(texts, batch_size=32)
np.save('data/chunk_embeddings.npy', embeddings)
```

When using the CLI, pass the path to the `.npy` file with `--embeddings data/chunk_embeddings.npy`.

Alternatively, we provide a handy script you can use to generate embeddings locally without managing the embedding model directly:

```pwsh
python tools/generate_embeddings.py --parquet data/chunks.parquet --output data/chunk_embeddings.npy --text-column text
```

## Format of `output/dedup_clusters.json`

Example output structure produced by the CLI:

```json
{
  "version": "1.0",
  "metadata": {
    "method": "hybrid",
    "timestamp": "2025-11-24T00:00:00Z"
  },
  "canonical_map": {
    "c1": "c1",
    "c2": "c1",
    "c3": "c3"
  },
  "clusters": {
    "0": ["c1", "c2"],
    "1": ["c3"]
  },
  "representatives": {
    "0": {"chunk_id": "c1", "score": 0.2, "cluster_size": 2},
    "1": {"chunk_id": "c3", "score": 0.6, "cluster_size": 1}
  }
}
```

## Recommended CI step

To ensure dedup functionality is covered in CI, add a small workflow that runs the dedup tests and the dedup integration smoke test (`test_integration_dedup_flow.py`). Example snippet for a GitHub Actions job:

```yaml
- name: Run dedup tests
  run: |
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    # For development and tests, prefer the dev extra:
    pip install -e '.[dev]'
    # Or use the repo copy: pip install -r requirements/requirements-dev.txt
    pytest tests/deduplication -k "deduplicator or end_to_end_dedup_and_index_flow" -q
```

This keeps the dedup pipeline checked for regressions without running heavier FAISS-only tests on every PR.


- If `DocumentRetriever` doesn't load a dedup map, check `config.deduplication.map_path` and run the CLI manually to generate a map.
- If using HDBSCAN/UMAP on Windows and facing installation issues, run the `minhash` or `sklearn` fallback mode with the `--method` flag.

## License & Attribution

Deduplication algorithms use open-source libraries (FAISS, HDBSCAN, UMAP, datasketch/MinHash). Please follow the libraries' respective licenses.
