# BEIR Benchmark Adapter

This adapter allows you to run standard [BEIR](https://github.com/beir-cellar/beir) benchmarks using CUBO's full retrieval pipeline.

## Quick Start

```bash
# Download BEIR dataset (e.g., FiQA)
python tools/download_beir_subsets.py --subsets fiqa nfcorpus --output-dir data/beir_hf

# Run benchmark with reindexing
python tools/run_beir_adapter.py \
    --corpus data/beir/beir_corpus.jsonl \
    --queries data/beir/queries.jsonl \
    --index-dir results/beir_adapter_index \
    --reindex \
    --evaluate \
    --qrels data/beir/qrels/dev.tsv \
    --top-k 100

# Or with existing index (faster)
python tools/run_beir_adapter.py \
    --queries data/beir/queries.jsonl \
    --index-dir results/beir_adapter_index \
    --evaluate \
    --qrels data/beir/qrels/dev.tsv
```

## Expected Output

```
--- BEIR Evaluation Results ---
NDCG@10: 0.4523
Recall@100: 0.7891
Precision@10: 0.0812
```

## Metrics

| Metric | Description |
|--------|-------------|
| NDCG@K | Normalized Discounted Cumulative Gain |
| Recall@K | Fraction of relevant docs in top K |
| MRR | Mean Reciprocal Rank |

## Requirements

- BEIR package: `pip install beir`
- Pre-downloaded corpus in JSONL format
- QRELS ground truth file (TSV format)

## File Formats

### Corpus (JSONL)
```json
{"_id": "123", "title": "Document Title", "text": "Document content..."}
```

### Queries (JSONL)
```json
{"_id": "1", "text": "What is machine learning?"}
```

### QRels (TSV)
```
query-id	corpus-id	score
1	123	1
1	456	1
```

## Architecture

The adapter uses CUBO's production `DocumentRetriever` pipeline:
1. **Embedding**: SentenceTransformer model
2. **Dense Search**: HNSW index (fast, accurate)
3. **Hybrid**: BM25 + Dense fusion (configurable)
4. **Reranking**: Cross-encoder reranker (optional)

This ensures benchmark results reflect actual production performance.
