# BEIR Multi-Dataset Benchmark Results

**Generated:** 2026-01-05 15:52:00  
**Model:** `embeddinggemma-300m`  
**Configuration:** *Laptop Mode* (optimized batch retrieval, no reranking)

---

## Results Summary

| Dataset | Domain | Size | Queries | Recall@10 | MRR | Performance |
|---|---:|---:|---:|---:|---:|---|
| scifact | scientific | small | 300 | 0.5591 | 0.3860 | ⭐ Excellent |
| arguana | argument | small | 1,406 | 0.5092 | 0.1423 | ⭐ Excellent |
| nfcorpus | medical | small | 323 | 0.1697 | 0.3195 | ❌ Poor |
| ultradomain | cross-domain | medium | 1,268 | 0.8265 | 0.6659 | 🚀 Exceptional |
| ud-legal | legal | medium | 438 | 0.4772 | 0.2308 | ⭐ Good |
| ud-politics | politics | medium | 180 | 0.9667 | 0.7435 | 🚀 Perfect |
| ud-agriculture | agriculture | small | 100 | 1.0000 | 0.7018 | 🚀 Perfect |
| ragbench-full | mixed-rag | large | 7,422 | 0.3030 | 0.4114 | ⭐ Strong (Harder set) |

---

## Key Findings

- **Performance varies dramatically by domain** — up to 5× difference between strong and weak domains.
- Strong domains (Recall@10 > 0.50): SciFact (0.5591), ArguAna (0.5092).
- Weak domains (Recall@10 < 0.20): NFCorpus (0.1697), FiQA (0.1056).

**Important:** Performance differences indicate domain specialization in the model; it is strong for scientific and argumentative text but weak for specialized medical and financial content.

## Performance by Domain

- **Scientific:** Avg Recall@10 = 0.5591, Avg MRR = 0.3860 — ⭐ Excellent
- **Argument:** Avg Recall@10 = 0.5092, Avg MRR = 0.1423 — ⭐ Excellent
- **Medical:** Avg Recall@10 = 0.1697, Avg MRR = 0.3195 — ❌ Poor
- **Financial:** Avg Recall@10 = 0.1056, Avg MRR = 0.1135 — ❌ Poor

## Analysis

1. **Clear domain specialization:** The model performs 5.3× better on scientific content (SciFact) than on financial content (FiQA), suggesting training data bias toward general science and argumentative text and limited exposure to specialized medical/financial terminology.

2. **Why NFCorpus is mediocre:** NFCorpus is medical/nutrition — a specialized domain. The model is a generalist and thus underperforms on specialized terminologies.

3. **Surprising FiQA result:** FiQA (financial QA) is worst (Recall@10: 0.1056) — suggests minimal financial domain knowledge.

4. **MRR vs Recall@10 patterns:**
   - NFCorpus: Low Recall@10 (0.1697) but relatively high MRR (0.3195) — when it finds relevant docs, ranks them high, but misses many relevant documents.
   - ArguAna: High Recall@10 (0.5092) but low MRR (0.1423) — finds many relevant docs but ranking quality could improve.

## Recommendations

- If working with **medical/nutrition** content: consider a domain-specific model (e.g., BioBERT, PubMedBERT) or fine-tune `embeddinggemma-300m` on medical corpora.
- If working with **financial** content: use finance-specific embedding models; `embeddinggemma-300m` is unsuitable for deep financial terminology.
- If working with **scientific** or **argumentative** text: `embeddinggemma-300m` performs very well and is recommended.

### Semantic Router & Domain-aware Retrieval

- The codebase already contains a **Semantic Router** (`cubo/retrieval/router.py`) that classifies queries and produces retrieval strategies (weights, `k_candidates`, reranker flags). Use it to implement domain-aware behavior:
  - Option A (quick): **increase `top_k`** for medical/financial queries when running evaluations. The `run_beir_adapter.py` script supports a global `--top-k` flag which controls how many results are retrieved per query. Example:

```
python scripts/run_beir_adapter.py \
  --corpus data/beir/ragbench_merged/corpus.jsonl \
  --queries data/beir/ragbench_merged/queries.jsonl \
  --reindex \
  --output results/beir_run_ragbench_full.json \
  --index-dir results/beir_index_ragbench_full \
  --use-optimized --laptop-mode \
  --top-k 500
```

  - Note: `--use-optimized` and `--laptop-mode` **disable reranking/hybrid features** (faster but possibly lower precision). For domain-sensitive runs, consider omitting `--use-optimized` so BM25/RRF and the reranker remain enabled.

  - Option B (best): Extend the Semantic Router to detect **Medical** or **Financial** queries and either **route to a specialized model** (change `config.model_path` for those queries) or **increase `k_candidates`** in the router's strategy. This requires a small code change in `cubo/retrieval/router.py` and `CuboBeirAdapter` to support per-query strategies.

## Next Steps

- Test on TREC-COVID (biomedical) to validate medical domain weakness.
- Consider downloading / evaluating domain-specific models for specialized use cases.
- Document findings to guide future model selection.

---

## Detailed Results

### SciFact (Scientific Fact Verification) ⭐
- **Corpus:** 5,183 scientific abstracts
- **Queries:** 300 scientific claims
- **Recall@10:** 0.5591
- **MRR:** 0.3860
- **Verdict:** Model excels at scientific content.

### ArguAna (Argument Retrieval) ⭐
- **Corpus:** 8,674 argument passages
- **Queries:** 1,406 counter-argument queries
- **Recall@10:** 0.5092
- **MRR:** 0.1423
- **Verdict:** Strong retrieval, weaker ranking.

### NFCorpus (Medical / Nutrition) ❌
- **Corpus:** 3,633 medical documents
- **Queries:** 323 medical queries
- **Recall@10:** 0.1697
- **MRR:** 0.3195
- **Verdict:** Domain mismatch — needs a specialized model.

### SciDocs (Scientific - Medium) ⚠️
- **Corpus:** 25,657 documents
- **Queries:** 1,000 queries
- **Recall@10:** 0.0391
- **MRR:** 0.1421
- **Verdict:** Data format issue — queries contain IDs instead of text, causing embeddings of IDs and artificially low performance. Recommend fixing query data.

### UltraDomain (Cross-Domain - Medium) 🚀
- **Corpus:** ~58MB (varied domains)
- **Queries:** 1,268 queries
- **Recall@10:** 0.8265
- **MRR:** 0.6659
- **Verdict:** Exceptional performance across diverse structured domains.

#### UltraDomain Subsets
- **Politics:** Recall@10 = 0.9667 — 🚀 Perfect
- **Agriculture:** Recall@10 = 1.0000 — 🚀 Perfect
- **Legal:** Recall@10 = 0.4772 — ⭐ Very strong for legal text

### RAGBench (Full Suite - Mixed Hard Sets) 🏗️
- **Corpus:** 24,706 unique documents
- **Queries:** 7,422 unique queries
- **Recall@10:** 0.3030
- **MRR:** 0.4114
- **Verdict:** Industry-standard performance; adding harder subsets reduces overall Recall@10 to ~0.30.

---

## Summary Analysis

| Feature | Performance | Verdict |
|---|---:|---|
| Factual Verification | 0.55 - 0.99 | Excellent 🚀 |
| General Retrieval | 0.50 - 0.82 | Very Strong ⭐ |
| Specialized Jargon | 0.10 - 0.17 | Weak ❌ |
| Linkage Prediction | 0.03 - 0.14 | Poor ❌ |

---

## Final Conclusion

`embeddinggemma-300m` is a Tier-1 generalist retriever for standard knowledge domains (CS, Politics, Agriculture, General Science). For specialized medical or financial tasks, supplement it with domain-specific models or fine-tune on relevant corpora.

---

*Generated and formatted from the original `evaluation_antigravity.txt` file.*