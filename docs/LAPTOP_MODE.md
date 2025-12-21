# CUBO Laptop Mode Guide

## Overview

CUBO includes a **Laptop Mode** that automatically optimizes resource usage for systems with limited RAM (≤16GB) or CPU cores (≤6). This guide explains how it works and how to configure it.

## Quick Start

### Automatic Detection

Laptop mode is **auto-enabled** when:
- System RAM ≤ 16GB, OR
- CPU cores ≤ 6

You'll see this message in the logs:
```
Laptop mode auto-enabled based on system resources. Set CUBO_LAPTOP_MODE=0 to disable.
```

### Manual Control

**Force enable:**
```bash
python -m cubo --laptop-mode
# OR
set CUBO_LAPTOP_MODE=1  # Windows
export CUBO_LAPTOP_MODE=1  # Linux/Mac
```

**Force disable:**
```bash
python -m cubo --no-laptop-mode
# OR
set CUBO_LAPTOP_MODE=0  # Windows
export CUBO_LAPTOP_MODE=0  # Linux/Mac
```

## New: Hardware Optimizations

CUBO now includes advanced hardware detection to further improve performance on laptops:

- **CPU Tuning**: Automatically sets thread counts to match physical cores, preventing overheating and thrashing.
- **Quantized Models**: Automatically uses faster, smaller models if your CPU supports AVX2 instructions.

See [Hardware Optimization](HARDWARE_OPTIMIZATION.md) for technical details.

## What Laptop Mode Changes

### 1. Ingestion Optimizations

| Setting | Default | Laptop Mode | Impact |
|---------|---------|-------------|--------|
| `enrich_enabled` | `true` | `false` | Skips LLM chunk enrichment (biggest time saver) |
| `n_workers` | `4` | `1` | Single-threaded processing (reduces memory) |
| `batch_size` | `50` | `5` | Smaller batches (lower peak RAM) |
| `throttle_delay_ms` | `0` | `500` | Prevents CPU throttling |
| `auto_generate_scaffolds` | `true` | `false` | Skips document scaffolding |

### 2. Retrieval Optimizations

| Setting | Default | Laptop Mode | Impact |
|---------|---------|-------------|--------|
| `reranker_model` | `cross-encoder` | `null` | Uses semantic cache instead of cross-encoder |
| `semantic_cache.enabled` | `false` | `true` | Caches rerank results for repeated queries |
| `semantic_cache.threshold` | `0.92` | `0.92` | Minimum similarity for cache hit |

### 3. Storage Optimizations

| Setting | Default | Laptop Mode | Impact |
|---------|---------|-------------|--------|
| `document_cache_size` | `1000` | `500` | Smaller LRU cache for documents |
| `persist_embeddings` | `memory` | `npy_sharded` | Store embeddings on disk |
| `embedding_dtype` | `float32` | `float16` | 50% smaller embeddings |
| `embedding_cache_size` | `512` | `512` | LRU cache for hot embeddings |

### 4. Index Optimizations

| Setting | Default | Laptop Mode | Impact |
|---------|---------|-------------|--------|
| `hot_ratio` | `0.2` | `0.1` | Smaller hot index in memory |
| `promote_threshold` | `10` | `100` | Less aggressive promotion |
| `nlist` | `1024` | `512` | Simpler FAISS index |
| `pq_m` | `64` | `32` | Lower product quantization |

### 5. Deduplication Optimizations

| Setting | Default | Laptop Mode | Impact |
|---------|---------|-------------|--------|
| `max_candidates` | unlimited | `200` | Caps candidate pairs (prevents O(n²) memory) |

## Architecture Changes

### SQLite-Backed Document Store

Documents and metadata are stored in SQLite instead of in-memory dicts:

```
faiss_store/
├── documents.db      # SQLite with WAL mode
├── hot.faiss        # Hot FAISS index
├── cold.faiss       # Cold FAISS index
└── embeddings/      # Sharded numpy files (laptop mode)
    ├── embedding_index.json
    ├── embeddings_shard_0000.npy
    ├── embeddings_shard_0001.npy
    └── ...
```

Benefits:
- **Memory**: Only LRU-cached documents in RAM
- **Persistence**: Survives restarts without rebuild
- **Thread-safe**: WAL mode for concurrent access

### Sharded Embedding Storage

Embeddings are stored in numpy shards instead of memory:

```python
# Config options
vector_store.persist_embeddings = "npy_sharded"
vector_store.embedding_dtype = "float16"
vector_store.shard_size = 1000
```

Benefits:
- **Memory**: ~50% reduction with float16
- **Scalability**: Load only needed shards
- **Caching**: LRU cache for hot embeddings

### Reranker LRU Cache

Query results and embeddings are cached:

```python
# Config options
retrieval.semantic_cache.enabled = true
retrieval.semantic_cache.threshold = 0.92
retrieval.semantic_cache.max_entries = 500
```

Benefits:
- **Speed**: Instant results for repeated queries
- **CPU**: No re-encoding of seen documents
- **Memory**: Bounded by max_entries

## Programmatic Control

### Check Laptop Mode Status

```python
from src.cubo.config import config

if config.is_laptop_mode():
    print("Running in laptop mode")
```

### Get Laptop Mode Config

```python
from src.cubo.config import Config

laptop_config = Config.get_laptop_mode_config()
print(laptop_config)
```

### Apply Laptop Mode Manually

```python
from src.cubo.config import config

# Force enable
config.apply_laptop_mode(force=True)

# Check if applied
print(config.is_laptop_mode())  # True
```

### Revert to Default Mode

```python
from src.cubo.config import config

# Revert to default configuration and clear laptop optimizations
config.apply_default_mode(force=True)
print(config.is_laptop_mode())  # False
```

### Monitor Cache Performance

```python
from src.cubo.rerank.reranker import get_reranker_cache

cache = get_reranker_cache()
print(cache.stats)
# {'query_hits': 42, 'query_misses': 10, 'hit_rate': 80.77, ...}
```

## Benchmarks

### Memory Usage (10K documents)

| Mode | Peak RAM | Steady-State RAM |
|------|----------|------------------|
| Default | 4.2 GB | 3.8 GB |
| Laptop | 1.4 GB | 1.1 GB |

### Query Latency (after warmup)

| Mode | First Query | Cached Query |
|------|-------------|--------------|
| Default | 450ms | 420ms |
| Laptop | 380ms | 12ms |

### Ingestion Time (1K documents)

| Mode | With Enrichment | Without |
|------|-----------------|---------|
| Default | 45 min | 8 min |
| Laptop | N/A | 12 min |

## Troubleshooting

### Still Running Out of Memory?

1. Reduce `document_cache_size` further:
   ```python
   config.set('document_cache_size', 100)
   ```

2. Force on-disk embedding storage:
   ```python
   config.set('vector_store.persist_embeddings', 'mmap')
   ```

3. Reduce shard size for smaller chunks:
   ```python
   config.set('vector_store.shard_size', 500)
   ```

### Queries Too Slow?

1. Increase `document_cache_size`:
   ```python
   config.set('document_cache_size', 1000)
   ```

2. Use in-memory embeddings if you have RAM:
   ```python
   config.set('vector_store.persist_embeddings', 'memory')
   ```

3. Enable cross-encoder reranking if quality matters:
   ```python
   config.set('retrieval.reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
   ```

### Duplicates Not Detected?

The `max_candidates=200` cap may miss some duplicates in very similar corpora. Increase if needed:

```python
config.set('deduplication.max_candidates', 500)
```

## Related Documentation

- [Processing Guide](PROCESSING.md) - Document processing pipeline
- [Resource Optimization Plan](resource_optimization_plan.md) - Original analysis
- [Config Reference](../config.json) - All configuration options

## CLI and Script Options

Start the API server in a given mode using the new `--mode` flag:

```bash
# Start API server in laptop mode (sets CUBO_LAPTOP_MODE=1)
python start_api_server.py --mode laptop

# Start full stack in laptop mode and pass a specific config file
python scripts/start_fullstack.py --mode laptop --config-path configs/config_local.json
```

You can also pass `--dry-run` to `start_api_server.py` to print the effective env vars and exit (useful when writing scripts or CI checks):

```bash
python start_api_server.py --mode laptop --config configs/config_local.json --dry-run
```
