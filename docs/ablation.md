# Ablation & System Evaluation Suite

This folder contains scripts to run ablation experiments (dense-only, BM25-only, hybrid), measure reranker effect (gain vs memory), and collect system metrics (indexing time, peak RAM, latency p50/p95).

Quick commands:

- Run ablation for `nfcorpus` at top-k 50:

```
make ablation
```

- Measure reranker effect:

```
make reranker
```

- Run system metrics (indexing + query latency):

```
make system
```

- Append results to `evaluation_antigravity.md`:

```
make update-report
```

Notes:
- The workflow is designed for manual dispatch in CI and is not run automatically on every PR.
- For heavy datasets, prefer running locally with enough RAM and CPU; use `--limit` where supported for faster smoke runs.
