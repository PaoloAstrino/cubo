# Deduplication - Maintainers Guide

This short README is for maintainers working on CUBO's deduplication feature.

Where configuration defaults live
- `src/cubo/config.py` contains the `deduplication` dictionary used by the CLI and retriever. Properties include:
  - `enabled`, `method`, `run_on`, `representative_metric`, `similarity_threshold`, `map_path`
  - `prefilter`, `ann`, `clustering` sub-objects

Extending the pipeline
- Adding a new ANN backend: implement `_your_backend_neighbors(self, embeddings, k)` and add the lookup in `_query_ann_neighbors` in `HybridDeduplicator` (`src/cubo/deduplication/semantic_deduplicator.py`).
- Adding a new clustering algorithm: implement an internal method with the same signature as `_cluster_with_hdbscan`, and add the selection in `find_clusters`.
- Representative selection rules: the `select_representatives` function picks the best representative based on `representative_metric`. To add a new metric, extend `_extract_metric` in `HybridDeduplicator`.

Notes on tests
- Unit tests live in `tests/deduplication/` and are designed to avoid heavy native dependencies where possible. We use deterministic FakeEmbedder fixtures to ensure test stability.
- Integration test `test_integration_dedup_flow.py` performs: `deduplicate` (CLI) -> `_filter_to_representatives` -> `DocumentRetriever` map loading. Ensure the test keeps small data (4+ chunks) to remain CI friendly.

Performance concerns
- The pipeline can be heavy when running on full datasets; prefer to test on small sample datasets for CI.
