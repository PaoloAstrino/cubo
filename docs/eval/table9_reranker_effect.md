# Reranker Effect Study — Table 9 Data

Based on NFCorpus benchmark (from `evaluation_antigravity.md` Section 1).

| Configuration | Recall@10 | nDCG@10 | MRR | Peak RAM | Latency (p95) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **Laptop Mode** (Dense Only) | **0.1697** | **0.1798** | **0.3195** | **< 8 GB** | **< 5 ms** | ✅ Selected |
| **Server Mode** (Hybrid + Rerank) | ~0.22 (+29%) | ~0.23 | ~0.37 | **> 16 GB** | > 250 ms | ❌ Too Heavy |

## Key Findings
- **ΔRecall@10:** +0.05 absolute (+29% relative improvement with reranker)
- **Memory Cost:** >16GB peak RAM (doubles memory usage)
- **Latency Cost:** 50x slower (>250ms vs <5ms per query)
- **Decision:** Reranker disabled by default in Laptop Mode to preserve speed and accessibility while maintaining competitive accuracy on general domains (0.50+ Recall on SciFact, ArguAna, UltraDomain).

## Justification
The reranker provides modest gains on weak domains (NFCorpus medical: +5% Recall) but at extreme hardware cost. For the target audience (developers with <16GB RAM), dense-only retrieval is the optimal trade-off.

---

**Source:** evaluation_antigravity.md, Section "Reranker Ablation (Dense-only vs Hybrid+Rerank)"  
**Date:** 2026-01-06
