# CUBO Performance Testing Infrastructure

Comprehensive performance testing infrastructure for comparing CUBO against published benchmarks (e.g., LightRAG). Tests include IR metrics (Recall@K, nDCG@K), latency profiling (p50/p95/p99), memory usage (RAM/VRAM), ingestion throughput, compression ratios, and RAGAS-style evaluation.

## Overview

This testing infrastructure allows reproducible performance measurement across:
- **Retrieval Quality**: Recall@K, nDCG@K, Precision@K, MRR
- **Latency**: p50/p95/p99 per-query latency, retrieval vs generation breakdown
- **Memory**: RAM and VRAM usage (peak/average)
- **Ingestion**: Throughput (GB/minute), compression ratio, indexing time
- **RAG Quality**: Answer relevance, groundedness, faithfulness (when LLM judge available)

## Components

### 1. Performance Utilities (`src/cubo/evaluation/perf_utils.py`)

Core measurement utilities:
- `sample_latency()` - Measure function latency with percentile statistics
- `sample_memory()` - Track RAM/VRAM usage over time
- `log_hardware_metadata()` - Capture CPU, GPU, RAM, CUDA version, git commit

### 2. IR Metrics (`src/cubo/evaluation/metrics.py`)

Information Retrieval metrics:
- `IRMetricsEvaluator` - Compute Recall@K, Precision@K, nDCG@K, MRR
- `GroundTruthLoader` - Load BeIR-format or custom ground truth files

### 3. RAG Testing Framework (`scripts/run_rag_tests.py`)

Main test orchestrator with three modes:
- **Full RAG**: Complete retrieval + generation + evaluation
- **Retrieval-only**: IR metrics without generation (isolates retrieval quality)
- **Ingestion-only**: Data loading and indexing only

### 4. Ingestion Throughput Testing (`scripts/test_ingestion_throughput.py`)

Measures:
- Ingestion time (seconds/GB, GB/minute)
- Memory consumption during ingestion
- Compression ratio (raw data vs stored index size)
- Chunks processed per second

## Quick Start

### Run Full RAG Tests with IR Metrics

```powershell
# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run with ground truth for IR metrics
python scripts/run_rag_tests.py `
    --questions test_questions.json `
    --data-folder data/ultradomain_sample `
    --ground-truth ground_truth/ultradomain_qrels.json `
    --mode full `
    --k-values 5,10,20 `
    --output results/full_rag_test.json
```

### Run Retrieval-Only Tests (No LLM Required)

```powershell
# Isolate retrieval quality without generation
python scripts/run_rag_tests.py `
    --questions test_questions.json `
    --data-folder data/ultradomain_sample `
    --ground-truth ground_truth/ultradomain_qrels.json `
    --mode retrieval-only `
    --k-values 5,10,20 `
    --output results/retrieval_only_test.json
```

### Test Ingestion Throughput

```powershell
# Measure ingestion performance
python scripts/test_ingestion_throughput.py `
    --data-folder data/ultradomain_sample `
    --output results/ingestion_test.json
```

## CLI Arguments

### `run_rag_tests.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--questions` | Path to questions JSON file | `test_questions.json` |
| `--data-folder` | Path to documents folder | `data` |
| `--ground-truth` | Path to ground truth file (BeIR or custom JSON) | None |
| `--mode` | Test mode: `full`, `retrieval-only`, `ingestion-only` | `full` |
| `--k-values` | Comma-separated K values for IR metrics | `5,10,20` |
| `--easy-limit` | Limit number of easy questions | None |
| `--medium-limit` | Limit number of medium questions | None |
| `--hard-limit` | Limit number of hard questions | None |
| `--output` | Output JSON filename | `test_results.json` |

### `test_ingestion_throughput.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-folder` | Path to documents folder (required) | - |
| `--output` | Output filename | `ingestion_results_{timestamp}.json` |
| `--fast-pass` | Use fast-pass ingestion | True |
| `--deep-ingest` | Use deep ingestion (overrides fast-pass) | False |

## Output Format

### Test Results JSON Schema

```json
{
  "metadata": {
    "test_run_timestamp": 1700000000.0,
    "mode": "full",
    "total_questions": 50,
    "success_rate": 0.96,
    "total_processing_time": 125.5,
    "avg_recall_at_k_5": 0.89,
    "avg_recall_at_k_10": 0.94,
    "avg_ndcg_at_k_10": 0.91,
    "avg_answer_relevance": 0.87,
    "avg_groundedness": 0.92,
    "hardware": {
      "cpu": {"model": "...", "cores_physical": 8},
      "ram": {"total_gb": 32.0},
      "gpu": {"device_name": "RTX 4050", "vram_total_gb": 6.0},
      "git": {"commit_hash": "abc123..."}
    }
  },
  "results": {
    "easy": [...],
    "medium": [...],
    "hard": [...]
  }
}
```

### Per-Query Result

```json
{
  "question": "What is...",
  "question_id": "easy_1",
  "difficulty": "easy",
  "retrieved_ids": ["doc1", "doc2", "doc3"],
  "retrieval_latency": {
    "p50_ms": 45.2,
    "p95_ms": 78.3,
    "mean_ms": 52.1
  },
  "ir_metrics": {
    "recall_at_k": {5: 1.0, 10: 1.0},
    "ndcg_at_k": {5: 0.95, 10: 0.94},
    "mrr": 1.0
  },
  "memory": {
    "ram_peak_gb": 2.3,
    "vram_peak_gb": 1.8
  },
  "response": "...",
  "processing_time": 1.25,
  "success": true
}
```

## Ground Truth Format

### Custom JSON Format

```json
{
  "question_id_1": ["relevant_doc_1", "relevant_doc_2"],
  "question_id_2": ["relevant_doc_3", "relevant_doc_4"]
}
```

### BeIR TSV Format

```
query-id\tcorpus-id\tscore
q1\tdoc1\t2
q1\tdoc2\t1
q2\tdoc3\t2
```

## Datasets

### Supported Datasets

1. **UltraDomain** (428 textbooks, 4 domains: Agriculture, CS, Legal, Mix)
   - Used by LightRAG paper
   - ~125 QA per domain
   - Best for comprehensiveness and diversity metrics

2. **BeIR** (FiQA, NFCorpus for health/legal domains)
   - Standard IR benchmark
   - Zero-shot retrieval evaluation
   - Provides nDCG baselines

3. **RAGBench** (100k multi-domain examples)
   - Large-scale evaluation
   - Faithfulness and groundedness focus

4. **Custom** (Enron/Wikipedia subsets for privacy testing)
   - Local evaluation
   - Privacy-preserving workflows

## Metrics Comparison with LightRAG

### Target Metrics (from how_to_compare.txt)

| Metric | LightRAG | CUBO Target | Test Command |
|--------|----------|-------------|--------------|
| Win Rate RAGAS | 60-85% | 65-90% | `--mode full` with judge LLM |
| Recall@10 | ~80-85% (est) | >90% | `--mode retrieval-only` |
| nDCG@10 | Not reported | >0.90 | `--mode retrieval-only` |
| Latency p50 | ~1-3s (est) | <200ms | All modes (automatic) |
| Latency p95 | Not reported | <800ms | All modes (automatic) |
| Memory (VRAM) | >32GB (est) | <6GB | All modes (automatic) |
| Ingestion | Hours (graph rebuild) | 2-3 min/GB | `test_ingestion_throughput.py` |
| Compression | N/A | 10:1 | `test_ingestion_throughput.py` |

## Reproducibility

### Fixed Configuration for Fair Comparison

1. **Embedding Model**: Fix to BGE or embeddinggemma-300m
2. **LLM**: Fix to Phi-3 for generation (if comparing generation)
3. **Seed**: Set via config for deterministic runs
4. **Hardware**: Log via `log_hardware_metadata()` automatically

### Example Reproducible Run

```powershell
# Set seed in config.json
# "seed": 42

# Run with fixed models
python scripts/run_rag_tests.py `
    --questions ultradomain_questions.json `
    --data-folder data/ultradomain_50gb `
    --ground-truth ultradomain_qrels.json `
    --mode retrieval-only `
    --k-values 5,10,20 `
    --output results/cubo_ultradomain_retrieval.json
```

## Next Steps

### Phase II - Benchmark Orchestration

Coming soon:
- `scripts/benchmark_runner.py` - Sweep datasets, configs, ablations
- `scripts/plot_results.py` - Generate comparison plots
- Ablation configs (hot/cold on/off, reranker on/off, BM25 weights)
 
#### Using `scripts/benchmark_runner.py` (Phase II)

The new `benchmark_runner.py` automates dataset/config/ablation sweeps and produces per-run JSON plus a `summary.csv` suitable for plotting. It integrates `test_ingestion_throughput.py` to include ingestion metrics in each run when requested.

Example quick run:

```powershell
python scripts/benchmark_runner.py `
  --datasets data/ultradomain_small:ultradomain `
  --configs configs/benchmark_config.json `
  --ablations configs/benchmark_ablations.json `
  --k-values 5,10,20 `
  --mode retrieval-only `
  --output-dir results/benchmark_runs
  --max-retries 3 `
  --retry-backoff 2.0
```

Note: The benchmark runner supports resuming long runs by skipping runs that already have results. Use `--skip-existing` to skip runs that have `benchmark_run.json` in the run directory. Use `--force` to remove and overwrite an existing run directory if you want to re-run the same test.

The config JSON format uses an array of objects with `name` and `config_updates` allowing programmatic application of `src.cubo.config.Config.update()`.

Example `configs/benchmark_config.json` (already included in repo):

```json
{
  "configs": [
    {"name": "hybrid_default", "config_updates": {"vector_store_backend": "faiss"}},
    {"name": "bm25_only", "config_updates": {"routing": {"factual_bm25_weight": 1.0}}}
  ]
}
```

Phase II will provide more advanced orchestration (parallelism, gitlab/gh runner integration) for massive runs.

### Phase III - CI & Validation

- CI smoke test (5 docs, 5 questions, no GPU)
- Unit tests for metric computation
- Schema validation for results JSON

### Phase IV - Plotting & Visualization

- `scripts/plot_results.py` - Reads `results/benchmark_runs/summary.csv` and the per-run `benchmark_run.json` files and generates the Part 2 graphs (win rate, latency vs size, Recall@K, memory vs size, ingestion time, compression vs recall). Example usage:

```powershell
python scripts/plot_results.py --results-dir results/benchmark_runs --output-dir results/plots
```

### JSON Schema Validation

Benchmark runner saves the `benchmark_run.json` files in each run folder. We provide a schema (`schemas/benchmark_output_schema.json`) and a validator helper script `src/cubo/evaluation/validate_schema.py` for schema validation.

Example validator run:

```powershell
python src/cubo/evaluation/validate_schema.py --json-file results/benchmark_runs/run1/benchmark_run.json --schema-file schemas/benchmark_output_schema.json
```

If the `jsonschema` package is not installed, the validator will report as not available: `jsonschema not installed`.

## Usage Examples

### 1. Quick Smoke Test (No Ground Truth)

```powershell
python scripts/run_rag_tests.py `
    --data-folder data/sample `
    --easy-limit 5 `
    --mode full
```

### 2. Full Evaluation on UltraDomain Subset

```powershell
python scripts/run_rag_tests.py `
    --questions ultradomain_agriculture_125q.json `
    --data-folder data/ultradomain_agri `
    --ground-truth ultradomain_agri_qrels.json `
    --mode full `
    --k-values 5,10,20 `
    --output results/ultradomain_agriculture_full.json
```

### 3. Retrieval-Only Benchmark (Fast, No LLM)

```powershell
python scripts/run_rag_tests.py `
    --questions beir_fiqa_queries.json `
    --data-folder data/beir_fiqa `
    --ground-truth beir_fiqa_qrels.tsv `
    --mode retrieval-only `
    --k-values 1,5,10,20,100 `
    --output results/beir_fiqa_retrieval.json
```

### 4. Ingestion Throughput on 100GB Corpus

```powershell
python scripts/test_ingestion_throughput.py `
    --data-folder data/wikipedia_100gb `
    --fast-pass `
    --output results/ingestion_wikipedia_100gb.json
```

## Troubleshooting

### Issue: Ground truth not loading

**Solution**: Ensure ground truth file is valid JSON or TSV format. Use `GroundTruthLoader.load_custom_format()` for debugging.

### Issue: CUDA out of memory

**Solution**: Reduce batch size in config.json or use CPU-only mode by unsetting CUDA environment.

### Issue: Missing document IDs in retrieval results

**Solution**: Ensure retriever returns documents with 'id', 'doc_id', or 'chunk_id' field for IR metric calculation.

## Contact & Contributing

For questions about the testing infrastructure or to contribute new metrics/datasets, see the main CUBO README.
