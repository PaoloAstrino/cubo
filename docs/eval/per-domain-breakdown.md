# Per-Domain Breakdown: Recall@10 Performance

> Canonical SOT: `docs/eval/evaluation_antigravity.md`

This table summarizes retrieval performance (Recall@10) across different domains and datasets, highlighting where the `embeddinggemma-300m` model excels and where domain-specific limitations exist.

## Summary Table

| Dataset | Domain | Queries | Recall@10 (Dense) | nDCG@10 | Verdict |
|---------|--------|---------|-------------------|---------|---------|
| **UltraDomain - Agriculture** | General/Agriculture | 100 | **1.0000** ⭐ | 0.7749 | Perfect retrieval |
| **UltraDomain - Politics** | General/Politics | 180 | **0.9667** ⭐ | 0.7976 | Near-perfect |
| **ArguAna (topk50)** | Argument/Debate | 1,406 | **0.8962** ⭐ | 0.4702 | Excellent retrieval |
| **UltraDomain - Medium** | Mixed Domains | 1,268 | **0.8265** ⭐ | 0.7049 | Strong cross-domain |
| **SciFact** | Scientific | 300 | **0.5591** ✅ | 0.4206 | Good scientific content |
| **FiQA (topk50 dense)** | Financial QA | 648 | **0.5244** ✅ | 0.4473 | Solid financial performance |
| **ArguAna (base)** | Argument/Debate | 1,406 | 0.5092 | 0.2287 | Baseline (lower top-k) |
| **UltraDomain - Legal** | Legal | 438 | 0.4772 | 0.2884 | Moderate legal domain |
| **RAGBench Merged** | Mixed RAG Tasks | 4,737 | 0.4146 | 0.4075 | Acceptable RAG performance |
| **NFCorpus** | Medical/Nutrition | 323 | 0.1697 ❌ | 0.1798 | Weak medical domain |
| **SciDocs** | Scientific (ID issue) | 1,000 | 0.0391 ⚠️ | 0.0749 | Data format issue |

## Key Insights

### 🚀 **Strong Domains** (Recall@10 > 0.80)
- **General/Structured Content:** UltraDomain subsets (Agriculture 1.0, Politics 0.97, Medium 0.83) show exceptional retrieval.
- **Argumentative Text:** ArguAna achieves 0.90 recall with topk50 dense retrieval, indicating strong performance on debate/argument structure.

### ✅ **Acceptable Domains** (Recall@10: 0.40–0.80)
- **Scientific Content:** SciFact (0.56) demonstrates solid retrieval for general scientific abstracts.
- **Financial QA:** FiQA (0.52 dense) performs better than expected, suggesting reasonable financial terminology coverage.
- **Legal:** UltraDomain Legal (0.48) shows moderate performance, acceptable for general legal queries.
- **Mixed RAG Tasks:** RAGBench (0.41) indicates acceptable baseline for varied retrieval scenarios.

### ❌ **Weak Domains** (Recall@10 < 0.20)
- **Medical/Nutrition:** NFCorpus (0.17) reveals significant domain mismatch — specialized medical terminology is underrepresented in the embedding model's training data.
- **SciDocs (Data Issue):** Ultra-low recall (0.04) is due to query data containing document IDs instead of text, not a model limitation.

## Recommendations

1. **Deploy as-is for:** General content, structured domains (agriculture, politics, mixed), argumentative text, and general scientific content.
2. **Augment for:** Legal and financial domains — consider domain-specific fine-tuning or hybrid retrieval with specialized BM25 indices.
3. **Replace for:** Medical/nutrition — use a specialized biomedical embedding model (e.g., PubMedBERT-based embeddings) for better recall.
4. **Fix data:** SciDocs requires query text resolution before evaluation.

## Ablation Comparison: Dense vs BM25

| Dataset | Dense Recall@10 | BM25 Recall@10 | Winner | Δ Recall |
|---------|-----------------|----------------|--------|----------|
| ArguAna (topk50) | **0.8962** | 0.8862 | Dense | +0.01 |
| FiQA (topk50) | **0.5244** | 0.5131 | Dense | +0.01 |
| NFCorpus (topk50) | **0.1697** | 0.1144 | Dense | +0.05 |

**Conclusion:** Dense embeddings consistently outperform BM25-only across all tested datasets, with the largest advantage in specialized domains (NFCorpus +49% relative improvement).
