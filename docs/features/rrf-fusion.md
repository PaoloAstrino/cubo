# Reciprocal Rank Fusion (RRF)

> **Category:** Feature | **Status:** Active

Technical details on how RRF combines dense and sparse retrieval scores for robust hybrid search.

---

CUBO uses **Reciprocal Rank Fusion (RRF)** to combine semantic (dense) and BM25 (sparse) retrieval results.

## Why RRF?

RRF is more robust than linear weighted scoring because:

1. **Score-agnostic**: Uses rank positions instead of raw scores
2. **Distribution invariant**: Works regardless of score scale differences between retrievers
3. **Proven effectiveness**: Standard in hybrid search systems (Elasticsearch, Pinecone, etc.)

## How It Works

```
RRF_score(doc) = Σ 1/(k + rank_i(doc))  for each retrieval method i
```

Where `k` is a constant (default: 60) that dampens the impact of high-ranking documents.

### Example

| Document | Semantic Rank | BM25 Rank | RRF Score |
|----------|---------------|-----------|-----------|
| doc_A    | 1             | 3         | 1/61 + 1/63 = 0.0322 |
| doc_B    | 2             | 1         | 1/62 + 1/61 = 0.0325 |
| doc_C    | 3             | 2         | 1/63 + 1/62 = 0.0320 |

Result: doc_B ranks highest despite not being #1 in either individual ranking.

## Configuration

RRF is used by default in the retrieval pipeline. The `k` constant can be adjusted:

```python
# In strategy.py
self.retrieval_strategy.combine_results_rrf(semantic, bm25, top_k, rrf_k=60)
```

- **Lower k** (e.g., 20): More emphasis on top-ranked documents
- **Higher k** (e.g., 100): Smoother blending of ranks

## Code Location

- Implementation: [`cubo/retrieval/fusion.py`](../cubo/retrieval/fusion.py)
- Strategy wrapper: [`cubo/retrieval/strategy.py`](../cubo/retrieval/strategy.py)
- Usage: [`cubo/retrieval/retriever.py`](../cubo/retrieval/retriever.py)

## References

- [Original RRF Paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
