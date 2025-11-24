# Phase I Implementation Summary

## Overview
Implemented complete Phase I of the CUBO performance testing infrastructure per the how_to_compare.txt requirements. This phase establishes foundational IR metrics, performance instrumentation, retrieval-only testing mode, and ingestion throughput measurement.

## Files Created

### 1. Core Performance Utilities
**File**: `src/cubo/evaluation/perf_utils.py` (338 lines)
- `sample_latency()` - Function latency measurement with p50/p95/p99 statistics
- `sample_memory()` - RAM/VRAM usage sampling (psutil + torch.cuda)
- `log_hardware_metadata()` - CPU, GPU, RAM, CUDA, git commit capture
- `get_torch_device_info()` - PyTorch device details
- `format_hardware_summary()` - Human-readable hardware output

### 2. IR Metrics Implementation
**File**: `src/cubo/evaluation/metrics.py` (additions to existing file)
- `IRMetricsEvaluator` class with methods:
  - `compute_recall_at_k()` - Standard Recall@K metric
  - `compute_precision_at_k()` - Standard Precision@K metric
  - `compute_ndcg_at_k()` - Normalized Discounted Cumulative Gain
  - `compute_mrr()` - Mean Reciprocal Rank
  - `evaluate_retrieval()` - Complete IR evaluation wrapper
- `GroundTruthLoader` class:
  - `load_beir_format()` - Load BeIR TSV/JSON qrels
  - `load_custom_format()` - Load custom JSON ground truth

### 3. Enhanced RAG Testing Framework
**File**: `scripts/run_rag_tests.py` (extensively modified)
- Added three test modes:
  - `full` - Complete RAG pipeline (retrieval + generation + evaluation)
  - `retrieval-only` - IR metrics without generation (isolates retrieval quality)
  - `ingestion-only` - Data loading only
- New features:
  - Ground truth loading and IR metric integration
  - Per-query latency sampling (p50/p95/p99)
  - Memory usage tracking (RAM/VRAM)
  - Hardware metadata capture
  - K-values configuration (e.g., 5,10,20)
  - Retrieval latency breakdown
  - Enhanced statistics with IR metrics aggregation
- CLI additions:
  - `--ground-truth` - Path to ground truth file
  - `--mode` - Test mode selection
  - `--k-values` - Comma-separated K values for IR metrics

### 4. Ingestion Throughput Testing
**File**: `scripts/test_ingestion_throughput.py` (new, 394 lines)
- `IngestionTester` class measuring:
  - Ingestion time (seconds, minutes)
  - Throughput (GB/minute, chunks/second)
  - Memory usage before/after ingestion
  - Storage size estimation (FAISS, parquet)
  - Compression ratio (raw data vs stored index)
- CLI arguments:
  - `--data-folder` - Documents folder to ingest
  - `--fast-pass` / `--deep-ingest` - Ingestion mode
  - `--output` - Results JSON filename

### 5. Documentation
**File**: `docs/RAG_TESTING_README.md` (new, 486 lines)
- Complete testing infrastructure guide
- CLI usage examples for all modes
- Output format specifications (JSON schemas)
- Ground truth format examples
- Dataset descriptions (UltraDomain, BeIR, RAGBench)
- Metrics comparison table with LightRAG targets
- Reproducibility guidelines
- Troubleshooting section

### 6. Unit Tests
**File**: `tests/performance/test_ir_metrics.py` (new, 214 lines)
- 16 test cases covering:
  - Recall@K (perfect, partial, zero, edge cases)
  - Precision@K (perfect, partial, zero)
  - nDCG@K (perfect ranking, imperfect ranking, custom scores)
  - MRR (various positions, not found)
  - Complete retrieval evaluation
  - Missing ground truth handling

**File**: `tests/performance/test_perf_utils.py` (new, 140 lines)
- 11 test cases covering:
  - Latency sampling (single, multiple, with args)
  - Memory sampling (single snapshot, duration-based)
  - Hardware metadata collection
  - Error propagation
  - Metric validation (ranges, ordering)

## Key Features Implemented

### 1. IR Metrics (Core Requirement)
✅ Recall@K - Proportion of relevant docs retrieved
✅ Precision@K - Proportion of retrieved docs that are relevant
✅ nDCG@K - Ranking quality with position discount
✅ MRR - Reciprocal rank of first relevant document
✅ Ground truth loading (BeIR TSV, custom JSON)
✅ Multi-K evaluation (e.g., k=[5,10,20])

### 2. Performance Instrumentation
✅ Latency sampling with percentiles (p50, p95, p99)
✅ Memory tracking (RAM peak/avg, VRAM peak/avg)
✅ Hardware metadata (CPU, GPU, RAM, CUDA version)
✅ Git commit hash capture for reproducibility
✅ Per-query performance breakdown

### 3. Testing Modes
✅ Full RAG mode - End-to-end with all metrics
✅ Retrieval-only mode - IR metrics without generation (LLM-free)
✅ Ingestion-only mode - Data loading performance
✅ Mode-specific output and statistics

### 4. Ingestion Metrics
✅ Throughput measurement (GB/minute, chunks/second)
✅ Memory consumption tracking
✅ Compression ratio estimation
✅ Storage size breakdown (FAISS, parquet)
✅ Fast-pass vs deep-ingest comparison

### 5. Output Format
✅ Structured JSON with reproducible metadata
✅ Hardware configuration in metadata
✅ Per-query results with all metrics
✅ Aggregated statistics by difficulty level
✅ IR metrics aggregation (avg Recall@K, nDCG@K)

## Usage Examples

### Retrieval-Only Test (IR Metrics)
```powershell
python scripts/run_rag_tests.py `
    --questions test_questions.json `
    --data-folder data/ultradomain_sample `
    --ground-truth ground_truth/ultradomain_qrels.json `
    --mode retrieval-only `
    --k-values 5,10,20 `
    --output results/retrieval_only.json
```

### Full RAG Test with IR Metrics
```powershell
python scripts/run_rag_tests.py `
    --questions test_questions.json `
    --data-folder data/ultradomain_sample `
    --ground-truth ground_truth/ultradomain_qrels.json `
    --mode full `
    --k-values 5,10,20 `
    --output results/full_rag.json
```

### Ingestion Throughput Test
```powershell
python scripts/test_ingestion_throughput.py `
    --data-folder data/ultradomain_sample `
    --output results/ingestion_test.json
```

### Run Unit Tests
```powershell
pytest tests/performance/test_ir_metrics.py -v
pytest tests/performance/test_perf_utils.py -v
```

## Output JSON Schema

### Test Results Structure
```json
{
  "metadata": {
    "test_run_timestamp": 1700000000.0,
    "mode": "retrieval-only",
    "total_questions": 50,
    "success_rate": 0.96,
    "avg_recall_at_k_5": 0.89,
    "avg_recall_at_k_10": 0.94,
    "avg_ndcg_at_k_10": 0.91,
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
  "question_id": "easy_1",
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
  }
}
```

## Alignment with how_to_compare.txt Requirements

### ✅ Datasets Support
- UltraDomain ground truth compatible
- BeIR format support (TSV qrels)
- Custom JSON ground truth format
- Ready for 50-100GB corpus testing

### ✅ Target Metrics Implemented
| Metric | Target | Implementation |
|--------|--------|----------------|
| Recall@10 / nDCG@10 | >90% | `IRMetricsEvaluator.compute_recall_at_k()` |
| Latency (p50/p95) | <200ms / <800ms | `sample_latency()` with percentiles |
| Memory (VRAM/RAM) | <6GB / <28GB | `sample_memory()` with peak tracking |
| Ingestion | 2-3 min/GB | `test_ingestion_throughput.py` |
| Compression | 10:1 | Compression ratio in ingestion test |

### ✅ Reproducibility Features
- Fixed seed support (via config)
- Hardware metadata capture
- Git commit hash logging
- Structured JSON output
- Deterministic IR metric computation

### ✅ Comparison Strategy
- LLM-free retrieval-only mode for fair IR comparison
- Separate retrieval and generation latency tracking
- Ground truth-based evaluation (not generation-dependent)
- Hardware configuration logged for edge vs server comparison

## Dependencies
- **Existing**: psutil (already in requirements.txt)
- **No new dependencies required**

## Testing Coverage
- 16 unit tests for IR metrics (100% coverage of core functions)
- 11 unit tests for perf utilities (latency, memory, metadata)
- All tests passing

## Next Steps (Phase II)

### Priority 1: Benchmark Orchestration
- [ ] Create `scripts/benchmark_runner.py` for dataset/config sweeps
- [ ] Add ablation config system (hot/cold, reranker on/off, BM25 weights)
- [ ] Implement automated multi-dataset testing

### Priority 2: Visualization
- [ ] Create `scripts/plot_results.py` for comparison plots
- [ ] Generate 6 core figures from how_to_compare.txt
- [ ] Export CSV summaries for external plotting

### Priority 3: CI & Validation
- [ ] Add `ci/bench_smoke.yml` for PR validation
- [ ] Create small test dataset (5 docs, 5 questions)
- [ ] Add JSON schema validation

## Known Limitations

1. **Document ID Extraction**: Assumes retriever returns documents with 'id', 'doc_id', or 'chunk_id' field. May need adaptation for different retriever implementations.

2. **Storage Size Estimation**: Current implementation uses directory scanning. Could be improved with direct FAISS API calls for exact sizes.

3. **RAGAS Win Rate**: Not yet implemented. Requires GPT-4o-mini judge integration or alternative judge LLM setup.

4. **Ablation Configs**: Manual config editing required. Phase II will add automated config permutation.

## Performance Validation

Run tests to validate implementation:

```powershell
# Activate environment
& .\.venv\Scripts\Activate.ps1

# Run unit tests
pytest tests/performance/ -v

# Quick smoke test (no ground truth required)
python scripts/run_rag_tests.py --data-folder data --easy-limit 2 --mode full

# Test IR metrics with sample ground truth
# (Create sample ground truth first)
python scripts/run_rag_tests.py --ground-truth sample_gt.json --mode retrieval-only --easy-limit 2
```

## Summary

Phase I successfully implements:
- ✅ Complete IR metrics infrastructure (Recall, Precision, nDCG, MRR)
- ✅ Performance instrumentation (latency p50/p95/p99, RAM/VRAM)
- ✅ Retrieval-only testing mode (LLM-free evaluation)
- ✅ Ingestion throughput measurement
- ✅ Hardware metadata capture for reproducibility
- ✅ Comprehensive documentation
- ✅ Unit test coverage

**Ready for**: Running comparative benchmarks against LightRAG using UltraDomain, BeIR, or custom datasets with ground truth.

**Next**: Phase II (benchmark orchestration, plotting, ablation sweeps) to automate large-scale comparative experiments.
