# Resource Optimization Plan for Local RAG on "Cheap Laptop"

## Executive Summary
The current codebase is designed for a server-grade environment with ample RAM and GPU resources. Running it on a resource-constrained laptop (e.g., 8GB/16GB RAM, weak/no GPU) will result in **Out-Of-Memory (OOM) crashes** and **unacceptable ingestion latency** (hours instead of minutes).

This document outlines the specific bottlenecks identified in the code and provides a step-by-step plan to resolve them.

---

## 1. Critical Bottlenecks

### 🔴 RAM Bottlenecks (Risk: OOM Crashes)

1.  **Vector Store (`src/cubo/retrieval/vector_store.py`)**
    *   **Issue:** `FaissStore` keeps **all** document text and metadata in Python dictionaries in RAM (`self._docs`, `self._metas`).
    *   **Impact:** 1GB of raw text can consume 2-4GB of RAM due to Python object overhead. This is the #1 cause of crashes for local RAG.
    *   **Code:**
        ```python
        70: self._docs: Dict[str, str] = {}
        71: self._metas: Dict[str, Dict] = {}
        ```

2.  **Deep Ingestion Accumulation (`src/cubo/ingestion/deep_ingestor.py`)**
    *   **Issue:** The `DeepIngestor` accumulates **every single chunk** in a list (`all_chunks`) before saving to disk at the very end.
    *   **Impact:** Processing a large dataset will steadily consume RAM until the process dies just before completion.
    *   **Code:**
        ```python
        688: all_chunks: List[Dict[str, Any]] = []
        ...
        748: def _save_results(...) # Only called at the end
        ```

3.  **Deduplication Graph (`src/cubo/deduplication/deduplicator.py`)**
    *   **Issue:** The `Deduplicator` builds a NetworkX graph of all candidate pairs in memory.
    *   **Impact:** For large datasets, the number of edges can grow quadratically, exploding memory usage.

### 🔴 GPU/Compute Bottlenecks (Risk: Extreme Latency)

1.  **Aggressive Enrichment (`src/cubo/ingestion/deep_ingestor.py`)**
    *   **Issue:** By default, `enrich_enabled=True` triggers an LLM call (Llama 3) for **every chunk** to generate summaries/keywords.
    *   **Impact:** On a laptop, Llama 3 might take 10s per chunk. 1,000 chunks = ~3 hours of processing.
    *   **Code:**
        ```python
        423: enriched_data = enricher.enrich_chunks(texts)
        ```

2.  **On-the-Fly Reranking (`src/cubo/rerank/reranker.py`)**
    *   **Issue:** `LocalReranker` re-encodes document text during every search query.
    *   **Impact:** Adds 1-2s latency per query on CPU.
    *   **Code:**
        ```python
        117: doc_emb = self.model.encode(doc_content, convert_to_tensor=False)
        ```

---

## 2. Detailed Implementation Plan

### Phase 1: Fix RAM Bottlenecks (CRITICAL - Do This First)

#### 1.1. Refactor `FaissStore` to Use SQLite Instead of In-Memory Dicts

**File:** `src/cubo/retrieval/vector_store.py`

**Problem Lines:**
- Line 70: `self._docs: Dict[str, str] = {}`
- Line 71: `self._metas: Dict[str, Dict] = {}`
- Line 94-95: `self._docs[doc_id] = document` and `self._metas[doc_id] = metadata`

**Changes Needed:**

1. **Add SQLite connection in `__init__`** (around line 65):
   ```python
   import sqlite3
   from functools import lru_cache

   # In __init__, replace dict initialization with:
   self._db_path = os.path.join(self.index_dir, 'documents.db')
   self._init_document_db()
   ```

2. **Create `_init_document_db` method**:
   ```python
   def _init_document_db(self):
       """Initialize SQLite database for document storage."""
       conn = sqlite3.connect(self._db_path)
       conn.execute('''
           CREATE TABLE IF NOT EXISTS documents (
               id TEXT PRIMARY KEY,
               content TEXT NOT NULL,
               metadata TEXT NOT NULL
           )
       ''')
       conn.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(id)')
       conn.commit()
       conn.close()
   ```

3. **Replace `add()` method** (lines 88-98):
   ```python
   def add(self, ids: List[str], embeddings: List[np.ndarray],
           documents: List[str], metadatas: List[Dict]) -> None:
       # Keep FAISS part as-is
       self.index_manager.add(ids, embeddings)

       # Replace dict storage with SQLite
       conn = sqlite3.connect(self._db_path)
       for doc_id, document, metadata in zip(ids, documents, metadatas):
           conn.execute(
               'INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)',
               (doc_id, document, json.dumps(metadata))
           )
       conn.commit()
       conn.close()
   ```

4. **Replace `get()` method** (lines 100-105):
   ```python
   @lru_cache(maxsize=1000)  # Cache frequently accessed docs
   def get(self, doc_id: str) -> Optional[Dict]:
       conn = sqlite3.connect(self._db_path)
       row = conn.execute(
           'SELECT content, metadata FROM documents WHERE id = ?', (doc_id,)
       ).fetchone()
       conn.close()

       if row:
           return {'document': row[0], 'metadata': json.loads(row[1])}
       return None
   ```

**Estimated RAM Savings:** ~80% reduction (4GB → 800MB for 1GB text corpus)

---

#### 1.2. Implement Streaming Saves in `DeepIngestor`

**File:** `src/cubo/ingestion/deep_ingestor.py`

**Problem Lines:**
- Line 688: `all_chunks: List[Dict[str, Any]] = []`
- Line 748: `self._save_chunks_parquet(...)` only called once at end

**Changes Needed:**

1. **Add batch counter in `__init__`** (around line 150):
   ```python
   self.chunk_batch_size = 50  # Flush to disk every 50 chunks
   self.temp_parquet_files = []
   ```

2. **Modify `_process_files_parallel`** (around line 685-690):
   ```python
   # BEFORE (line 688):
   # all_chunks: List[Dict[str, Any]] = []

   # AFTER:
   all_chunks: List[Dict[str, Any]] = []
   batch_counter = 0

   # Inside the loop where chunks are accumulated (around line 720):
   for chunk_batch in file_result.get('chunks', []):
       all_chunks.append(chunk_batch)
       batch_counter += 1

       # Flush to disk every N chunks
       if batch_counter >= self.chunk_batch_size:
           self._flush_chunk_batch(all_chunks, run_id)
           all_chunks.clear()  # Free memory
           batch_counter = 0
   ```

3. **Add new `_flush_chunk_batch` method**:
   ```python
   def _flush_chunk_batch(self, chunks: List[Dict], run_id: str):
       """Flush a batch of chunks to a temporary parquet file."""
       if not chunks:
           return

       temp_file = self.output_dir / f'temp_chunks_{run_id}_{len(self.temp_parquet_files)}.parquet'
       df = pd.DataFrame(chunks)
       df.to_parquet(temp_file, index=False)
       self.temp_parquet_files.append(temp_file)
       logger.info(f"Flushed {len(chunks)} chunks to {temp_file.name}")
   ```

4. **Update `_save_chunks_parquet`** to merge temp files:
   ```python
   def _save_chunks_parquet(self, chunks: List[Dict], output_path: Path):
       # If we have temp files, merge them
       if self.temp_parquet_files:
           # Save final batch
           if chunks:
               self._flush_chunk_batch(chunks, 'final')

           # Merge all temp parquet files
           all_dfs = [pd.read_parquet(f) for f in self.temp_parquet_files]
           merged_df = pd.concat(all_dfs, ignore_index=True)
           merged_df.to_parquet(output_path, index=False)

           # Clean up temp files
           for temp_file in self.temp_parquet_files:
               temp_file.unlink()
           self.temp_parquet_files.clear()
       else:
           # Fallback to original behavior
           df = pd.DataFrame(chunks)
           df.to_parquet(output_path, index=False)
   ```

**Estimated RAM Savings:** Prevents OOM on large datasets (caps RAM at ~500MB regardless of corpus size)

---

### Phase 2: Fix Compute/Latency Bottlenecks

#### 2.1. Disable LLM Enrichment by Default

**File:** `src/cubo/config.py`

**Problem Line:**
- Around line 65 in `_get_default_config()`:
  ```python
  "enrich_enabled": True,  # ← This causes 3+ hour ingestion times
  ```

**Change:**
```python
"enrich_enabled": False,  # Fast Pass mode for laptops
```

**Alternative:** Create a "laptop mode" config preset:
```python
# In config.py, add a new method:
@staticmethod
def get_laptop_mode_config():
    """Optimized config for resource-constrained laptops."""
    base = Config._get_default_config()
    base['ingestion']['deep']['enrich_enabled'] = False
    base['ingestion']['deep']['n_workers'] = 1
    base['ingestion']['deep']['batch_size'] = 5
    base['retrieval']['reranker_model'] = None  # Disable cross-encoder
    return base
```

---

#### 2.2. Cache Document Embeddings in Reranker

**File:** `src/cubo/rerank/reranker.py`

**Problem Line:**
- Line 117: `doc_emb = self.model.encode(doc_content, convert_to_tensor=False)`

**Changes:**

1. **Add LRU cache** (at top of class, around line 20):
   ```python
   from functools import lru_cache

   class LocalReranker:
       def __init__(self, model):
           self.model = model
           self._embedding_cache = {}  # doc_id -> embedding
   ```

2. **Modify `_score_query_document_pair`** (line 101-125):
   ```python
   def _score_query_document_pair(self, query_emb, document):
       # Check if document already has embedding
       doc_id = document.get('metadata', {}).get('chunk_id')

       if doc_id and doc_id in self._embedding_cache:
           doc_emb = self._embedding_cache[doc_id]
       else:
           # Compute embedding
           doc_content = document.get('document', '')
           doc_emb = self.model.encode(doc_content, convert_to_tensor=False)

           # Cache it for future queries
           if doc_id:
               self._embedding_cache[doc_id] = doc_emb

       # ... rest of scoring logic
   ```

**Estimated Speedup:** 10x faster reranking (from ~2s to ~200ms per query)

---

### Phase 3: Advanced Optimizations (Optional)

#### 3.1. Use MiniBatchKMeans for Clustering

**File:** `src/cubo/processing/clustering.py`

**Line:** 90-104 in `_cluster_kmeans()`

**Change:**
```python
from sklearn.cluster import MiniBatchKMeans  # Add to imports

def _cluster_kmeans(self, embeddings: np.ndarray, n_clusters: int):
    # Replace KMeans with MiniBatchKMeans for memory efficiency
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=256,  # Process in batches to save RAM
        n_init=3  # Reduce iterations
    )
    labels = kmeans.fit_predict(embeddings)
    return labels, n_clusters
```

**RAM Savings:** ~50% reduction during scaffold generation

---

## 3. Implementation Order (Recommended)

### Week 1: Critical Fixes
1. ✅ **Day 1-2:** Implement SQLite storage in `vector_store.py` (Section 1.1)
2. ✅ **Day 3-4:** Add streaming saves to `deep_ingestor.py` (Section 1.2)
3. ✅ **Day 5:** Disable enrichment by default (Section 2.1)

### Week 2: Performance Recovery
4. ✅ **Day 6-7:** Add reranker caching (Section 2.2)
5. ✅ **Day 8:** Test on real laptop with 1GB corpus
6. ✅ **Day 9:** Benchmark ingestion time (target: <5 min for 10 books)

### Week 3: Polish (Optional)
7. ⚪ Implement MiniBatchKMeans (Section 3.1)
8. ⚪ Profile and optimize any remaining hotspots

---

## 3. Recommended Configuration for "Cheap Laptop"

Create a `config_local.json` with these overrides:

```json
{
  "vector_store_backend": "faiss",
  "ingestion": {
    "deep": {
      "n_workers": 1,
      "batch_size": 5,
      "throttle_delay_ms": 500,
      "enrich_enabled": false,
      "auto_generate_scaffolds": false
    }
  },
  "retrieval": {
    "reranker_model": null,
    "semantic_cache": {
      "enabled": true
    }
  }
}
```
