# Evaluation Checklist — Antigravity

This checklist contains detailed, actionable steps for completing the evaluation: commands to run, verification steps, expected outputs, and artifacts to attach to the final PR.

---

## Current status
- [x] Aggregate partial results & validate — **COMPLETED** (see `results/aggregated_metrics.json` and `results/aggregated_metrics.csv`)
  - Deliverables: `results/aggregated_metrics.json`, `results/aggregated_metrics.csv`, `docs/eval/aggregated_metrics.md` (per-dataset + per-domain rows)
  - Quick check command: `ls results | grep beir_run | sed -n '1,200p'` and inspect `results/*.json` files
  - Est. time: 10–30 min

---

## Tasks (detailed steps)

### 1) Aggregate partial results & validate (Priority: High)
- Goal: produce a single, validated metrics table for all datasets/modes containing Recall@10, MRR@10, nDCG@10, queries_evaluated.
- Steps:
  1. Locate all run outputs: `results/beir_run_*_topk50_*.json` and metric files `*_metrics_k10.json`.
  2. For each run without a `_metrics_k10.json`, run `scripts/calculate_beir_metrics.py` (see Task 2).
  3. Merge metrics into a single JSON/CSV (`results/aggregated_metrics.json` and `results/aggregated_metrics.csv`). Use pandas for robust merging:
     - Example: `python - <<'PY'
import json,glob,pandas as pd
rows=[]
for f in glob.glob('results/*_metrics_k10.json'):
    rows.append(json.load(open(f)))
pd.DataFrame(rows).to_csv('results/aggregated_metrics.csv', index=False)
PY`
  4. Validate: assert all target datasets present and that `queries_evaluated` matches expected counts.
- Verification:
  - Run: `python -c "import pandas as pd; df = pd.read_csv('results/aggregated_metrics.csv'); print(df[['dataset','mode','recall_at_k']])"`
  - Acceptance: Metrics exist for all datasets listed in `docs/eval/per_domain_breakdown.md`.
- Est. time: 10–45 min
- Artifacts: `results/aggregated_metrics.json`, `results/aggregated_metrics.csv`, `docs/eval/aggregated_metrics.md`


### 2) Compute missing secondary metrics (MRR@10, nDCG@10)
- Goal: ensure MRR and nDCG are present for every (dataset, mode) pair.
- Per-dataset checklist (mark as each completes):
  - [x] **fiqa** — computed (`results/beir_run_fiqa_topk50_{dense,bm25,hybrid}_metrics_k10.json`)
  - [x] **scifact** — `results/beir_run_scifact.json` (metrics saved to `results/beir_run_scifact_metrics_k10.json`)
  - [x] **arguana** — `results/beir_run_arguana.json` (metrics saved to `results/beir_run_arguana_metrics_k10.json`)
  - [x] **ultradomain (all subsets)** — `results/beir_run_ultradomain_*.json` (metrics saved to `results/beir_run_ultradomain_*_metrics_k10.json`)
  - [ ] **ragbench_full** — `results/beir_run_ragbench_full*.json`
  - [x] **ragbench_merged** — `results/beir_run_ragbench_merged.json` (metrics saved to `results/beir_run_ragbench_merged_metrics_k10.json`)
  - [x] **scidocs** — `results/beir_run_scidocs.json` (metrics saved to `results/beir_run_scidocs_metrics_k10.json`)
  - [x] **nfcorpus (topk10 runs)** — `results/beir_run_nfcorpus_topk10_*.json` (metrics saved to `results/beir_run_nfcorpus_topk10_*_metrics_k10.json`)

- Steps:
  1. Find run files missing metrics: `python - <<'PY'
import glob, os
for r in glob.glob('results/beir_run_*_topk50_*.json'):
    m = r.replace('.json','_metrics_k10.json')
    if not os.path.exists(m):
        print('Missing metrics for', r)
PY`
  2. For each missing: `python scripts/calculate_beir_metrics.py --results <run_file> --qrels data/beir/<dataset>/qrels/test.tsv --k 10`
  3. Re-run aggregation step (Task 1) to include newly computed metrics.
- Verification: each `beir_run_<dataset>_topk50_<mode>_metrics_k10.json` exists and contains keys `recall_at_k`, `mrr`, `ndcg`.
- Est. time: ~30s–5min per dataset (depends on queries)
- Artifact: updated metric files in `results/`


### 3) Complete ablation runs where needed (Table 7) — Dense / BM25 / Hybrid
- Goal: for every dataset in scope, have three runs (dense, bm25, hybrid) produced with identical top-k and indexing settings for fair comparison.
- Steps:
  1. Identify datasets missing any of the 3 modes by scanning `results/`.
  2. Run ablation for missing modes (or re-run full ablation forcing reindex):
     - `python scripts/run_ablation.py --dataset <dataset> --top-k 50 --index-dir results --output-dir results [--force-reindex]`
  3. For large datasets, run on a machine with sufficient RAM and monitor disk usage.
- Verification:
  - Check presence of `results/beir_run_<dataset>_topk50_{dense,bm25,hybrid}.json` and their metric files.
  - Acceptance criteria: recall and MRR computed and merged in aggregation.
- Est. time: 1–3+ hours per dataset (depending on dataset size).
- Artifact: run JSONs in `results/` and entries in aggregation.
- Current status: SciFact ablation started (dense mode running as of 2026-01-09 12:24 local time); remaining datasets queued.


### 4) Reranker effect study (Table 9) — justify disabling reranker when cost is high — **COMPLETED**
- Goal: measure the benefit (ΔRecall@10) of reranking vs base retrieval and measure hardware cost (peak RAM, latency overhead).
- Status: NFCorpus reranker ablation data already collected and documented in `evaluation_antigravity.md`.
- Key findings:
  - ΔRecall@10: +0.05 absolute (+29% relative) with reranker on NFCorpus
  - Peak RAM: >16GB (doubles usage)
  - Latency: 50x slower (>250ms vs <5ms per query)
  - Decision: Reranker disabled by default in Laptop Mode
- Deliverable: `docs/eval/table9_reranker_effect.md` (consolidated from existing data)
- Est. time: already measured
- Artifact: `docs/eval/table9_reranker_effect.md`


### 5) System metrics: ingestion time, peak RAM, latency p50/p95 — **COMPLETED**
- Goal: produce reproducible system metrics per dataset on target hardware.
- Status: NFCorpus system metrics already collected and documented in `evaluation_antigravity.md`.
- Key metrics (NFCorpus, laptop mode):
  - Indexing: 114s (3,633 docs), ~32 docs/sec
  - Peak RAM (indexing): ~6.5 GB
  - Query latency p50: <1ms, p95: <5ms
  - Index size: 36MB (~10MB per 1k docs)
- Deliverable: `docs/eval/system_metrics_summary.md` and `results/system_metrics_summary.json`
- Est. time: already measured
- Artifacts: `docs/eval/system_metrics_summary.md`, `results/system_metrics_summary.json`


### 6) Per-domain breakdown table
- Goal: a clear markdown table showing Recall@10 per domain/dataset for quick interpretation and Figure/Table inclusion.
- Steps:
  1. From `results/aggregated_metrics.csv`, pivot by dataset (or domain) and mode.
  2. Draft `docs/eval/per_domain_breakdown.md` with a table and short comment lines interpreting high/low results.
- Example line for the table:
  - `| UltraDomain-Legal | 0.48 |`  
  - `| NFCorpus | 0.17 |`
- Verification: Ensure numbers match aggregated metrics and include queries_evaluated column.
- Est. time: 10–20 min


### 7) Write narrative & tables in `evaluation_antigravity.md` (Italian)
- Goal: produce final write-up with Table 7 (ablation), Table 9 (reranker), system metrics summary, per-domain takeaways, and an honest conclusion.
- Steps:
  1. Add Table 7 and Table 9 as markdown tables (with captions and short notes on experiment conditions).
  2. Add system metrics paragraph describing ingestion time, memory footprint, latency, and trade-offs.
  3. Add per-domain interpretation paragraph in Italian (concise—2–4 sentences).
- Verification: Peer review, cross-check numbers with `results/aggregated_metrics.csv`.
- Est. time: 1–2 hours
- Artifact: updated `evaluation_antigravity.md` in repo


### 8) Reproducibility & scripts
- Goal: make it easy to reproduce any experiment and run the full suite for a dataset non-interactively.
- Steps:
  1. Add `-Dataset` (or `--dataset`) parameter to `run_full_benchmarks.ps1` and `--yes`/`--force` flag to skip prompts.
  2. Add helper Make targets in `Makefile` or simple PS1 wrapper: `make results/fiqa` that runs ablation + metrics.
  3. Document exact commands in `docs/eval/README.md` with expected runtime and hardware notes.
- Verification: Run `.
un_full_benchmarks.ps1 -Dataset fiqa -Quick -Yes` and confirm no interactive prompts appear.
- Est. time: 30–60 min


### 9) Final review, tests & PR
- Goal: sanity-check all artifacts and open a PR with results, scripts, and `docs/` updates.
- Steps:
  1. Run tests: `pytest -q` and ensure no regression (or run only relevant test subset).
  2. Add artifacts to `results/` and update `docs/eval/*` files.
  3. Create a PR with a concise summary: what changed, key numbers, and actions to reproduce.
- Verification: PR contains files, tests pass, and changes are reviewable.
- Est. time: 30–60 min


---

## Notes
- Primary metric: **Recall@10** (already collected for 8+ datasets).  
- Secondary metrics: **MRR@10**, **nDCG@10** — ensure coverage and add missing entries.  
- Story arc: strong performance on structured and general domains (e.g., contracts/legal), weaker on ultra-specialized jargon domains (document this honestly).

---

Place this file at the repository root to keep the team aligned and to track progress.
