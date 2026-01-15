# Local RAG Architectural Optimization Guide

This document serves as the technical "Source of Truth" for the architectural decisions and optimizations implemented in the CUBO RAG system. These changes ensure top-tier performance, stability, and resource efficiency for a 100% offline document assistant running on personal hardware.

---

## 1. Async Concurrency Control & Event Loop Integrity
**Status:** ✅ Implemented

*   **The Issue:** The FastAPI server used `async def` endpoints that called blocking synchronous code (e.g., LLM inference or indexing). This blocked the single-threaded event loop, freezing the entire server for all users during a query.
*   **The Fix:** Wrapped all heavy compute calls in `run_in_threadpool` and implemented a global `asyncio.Lock` to strictly serialize RAM-heavy operations.
*   **Code Transformation:**
    ```python
    # After optimization
    from starlette.concurrency import run_in_threadpool
    async with compute_lock:
        results = await run_in_threadpool(cubo_app.retriever.retrieve_top_documents, ...)
    ```
*   **Cost:** 0 MB RAM.
*   **Benefit:** The server remains responsive to health checks and light requests while processing heavy tasks. RAM usage is strictly capped by ensuring only one heavy task runs at a time.

---

## 2. Reciprocal Rank Fusion (RRF) for Hybrid Retrieval
**Status:** ✅ Implemented

*   **The Issue:** The system previously used a naive linear weighted sum (e.g., `semantic * 0.7 + bm25 * 0.3`). This is fragile because BM25 scores are unbounded while vector similarities are 0-1, leading to one signal dominating the other.
*   **The Fix:** Implemented Reciprocal Rank Fusion (RRF) in the retrieval core to combine Sparse (BM25) and Dense (Vector) search results based on rank position rather than raw score.
*   **Formula:** `score(d) = sum(1 / (k + rank_i(d)))`
*   **Cost:** 0 MB RAM.
*   **Benefit:** Drastically more robust search results, especially when keyword matches and semantic meaning are at odds.

---

## 3. Multilingual "Europocentric" Tokenization
**Status:** ✅ Implemented

*   **The Issue:** Standard BM25 implementations use simple whitespace splitting, which fails for morphologically rich European languages (e.g., not matching "gatto" with "gatti").
*   **The Fix:** Integrated `SimplemmaLemmatizer` and `MultilingualTokenizer` using `nltk` Snowball stemmers. Relaxed language detection constraints to support 1-2 word queries.
*   **Cost:** Negligible (< 5 MB).
*   **Benefit:** High-quality keyword search for Italian, French, German, Spanish, and English out of the box.

---

## 4. Modern "Instruct" Chat Templating
**Status:** ✅ Implemented

*   **The Issue:** Sending raw text to modern local LLMs (Llama 3, Mistral) causes hallucinations and reasoning failures because they expect specific structural tokens (e.g., `<|start_header_id|>`).
*   **The Fix:** Implemented a `ChatTemplateManager` using Jinja2 templates to correctly wrap document context and user queries in the model's native format.
*   **Template Example (Llama 3):**
    ```text
    <|start_header_id|>user<|end_header_id|>
    Context: {{ context }}
    Question: {{ query }}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    ```
*   **Cost:** 0 MB RAM.
*   **Benefit:** Significant reduction in hallucinations and improved formatting consistency.

---

## 5. SQLite WAL (Write-Ahead Logging) Mode
**Status:** ✅ Implemented

*   **The Issue:** High concurrency (e.g., background ingestion while querying) caused `Database is locked` errors in the default SQLite mode.
*   **The Fix:** Enabled `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL` on database initialization.
*   **Cost:** Slight increase in temporary disk usage (`-wal` file). 0 MB RAM.
*   **Benefit:** Readers (Queries) do not block Writers (Ingestion), ensuring smooth UI performance during data preparation.

---

## 6. Enforced Float16 Embedding Pipeline
**Status:** ✅ Implemented

*   **The Issue:** Sentence Transformers output `float32` by default, using 4 bytes per dimension.
*   **The Fix:** Enforced casting to `float16` immediately after generation and during SQLite persistence. Implemented "Just-In-Time" casting to `float32` for FAISS CPU computation compatibility.
*   **Cost:** **Reduces RAM/Disk usage for vectors by 50%.**
*   **Benefit:** Doubles the capacity of the "Hot" RAM index on the same hardware without measurable accuracy loss.

---

## 7. True Real-Time Token Streaming
**Status:** ✅ Implemented

*   **The Issue:** The API waited for the full LLM response before sending JSON, resulting in long "cold" spinners for the user.
*   **The Fix:** Updated both Ollama and `llama_cpp` generators to use native iterators, yielding tokens via FastAPI `StreamingResponse`.
*   **Cost:** 0 MB RAM.
*   **Benefit:** Perceived latency drops from ~10s to < 500ms (Time to First Token).

---

## 8. Structure-Aware Hierarchical Chunking
**Status:** ✅ Implemented

*   **The Issue:** Naive chunking by character count splits paragraphs and tables mid-sentence, destroying semantic context.
*   **The Fix:** Utilized `HierarchicalChunker` to respect paragraph boundaries (`\n\n`) and document headers, ensuring chunks are semantically complete units.
*   **Cost:** 0 MB RAM.
*   **Benefit:** Better retrieval precision and cleaner context for the LLM to process.

---

## 9. Explicit Garbage Collection on Ingestion
**Status:** ✅ Implemented

*   **The Issue:** Python's garbage collector can be lazy, leading to RAM "creep" during large ingestion runs (e.g., 50GB+ of data).
*   **The Fix:** Manually triggered `gc.collect()` after every batch of chunks is saved to Parquet in `DeepIngestor`.
*   **Cost:** Minimal CPU overhead.
*   **Benefit:** Prevents Out-Of-Memory (OOM) crashes during hours-long ingestion jobs on 8GB/16GB laptops.

---

## 10. Proactive Model "Warm-up" Route
**Status:** ✅ Implemented

*   **The Issue:** The first query after startup takes an extra 10+ seconds because models must be loaded from disk to RAM/VRAM.
*   **The Fix:** Implemented a background warm-up thread that triggers a dummy inference on startup if the system has > 16GB RAM.
*   **Cost:** RAM is occupied earlier, but peak usage is unchanged.
*   **Benefit:** The first user interaction is instant, creating a "Professional" first impression.
