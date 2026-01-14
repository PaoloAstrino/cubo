# System Metrics Summary

System hardware metrics measured on NFCorpus benchmark (laptop mode, CPU-only).

| Metric | Value | Target |
|---|---:|---|
| **Indexing Speed** | ~32 docs/sec | ✓ CPU-friendly |
| **Peak Memory (Indexing)** | ~6.5 GB | ✓ < 16GB |
| **Query Latency (p50)** | < 1 ms | ✓ Instant |
| **Query Latency (p95)** | < 5 ms | ✓ Real-time |
| **Index Size on Disk** | ~36 MB (3,633 docs) | ✓ Compact (~10MB per 1k docs) |

## Performance Profile
- **Indexing:** 114 seconds for 3,633 documents (~32 docs/sec on CPU). Model loading dominates initial overhead.
- **Query Speed:** FAISS IVFFlat index delivers sub-millisecond retrieval. Batch retrieval optimization reduces per-query overhead to negligible levels.
- **Memory Footprint:** Peak RSS stays under 8GB during indexing and querying, comfortably within developer laptop constraints (16GB RAM, no GPU required).

## Extrapolated Estimates (other datasets)
Based on NFCorpus baseline:
- **FiQA (648 queries):** ~1ms p50, ~5ms p95 query latency
- **SciFact (1,109 queries):** ~1ms p50, ~6ms p95 query latency

> Canonical SOT: `docs/eval/evaluation_antigravity.md`
- **ArguAna (1,406 queries):** ~1ms p50, ~7ms p95 query latency
- **UltraDomain (1,268 queries):** ~1ms p50, ~6ms p95 query latency

(Actual measurements may vary based on corpus size and query complexity.)

---

**Source:** evaluation_antigravity.md, Section "System Hardware Metrics (Laptop Mode)"  
**Measured:** 2026-01-06  
**Consolidated:** 2026-01-09
