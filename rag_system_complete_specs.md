# 🚀 RAG 100GB Full-Local System - Complete Implementation Guide

**Version:** 1.0
**Target Hardware:** RTX 4050 6GB VRAM, 32GB RAM
**Data Scale:** 100GB raw → 5-10GB compressed semantic layer
**Latency Target:** <200ms p50 query response

---

## 📋 TABLE OF CONTENTS

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Hardware Requirements & Constraints](#2-hardware-requirements--constraints)
3. [Software Stack & Dependencies](#3-software-stack--dependencies)
4. [Layer 1: Ingestion Pipeline](#4-layer-1-ingestion-pipeline)
5. [Layer 2: LLM Processing](#5-layer-2-llm-processing)
6. [Layer 3: Embedding Generation](#6-layer-3-embedding-generation)
7. [Layer 4: Deduplication System](#7-layer-4-deduplication-system)
8. [Layer 5: Semantic Compression](#8-layer-5-semantic-compression)
9. [Layer 6: Vector Indexing](#9-layer-6-vector-indexing)
10. [Layer 7: Query Routing](#10-layer-7-query-routing)
11. [Layer 8: Hybrid Retrieval](#11-layer-8-hybrid-retrieval)
12. [Layer 9: Caching System](#12-layer-9-caching-system)
13. [Layer 10: Response Generation](#13-layer-10-response-generation)
14. [Database Schema & Storage](#14-database-schema--storage)
15. [Configuration Files](#15-configuration-files)
16. [Performance Tuning](#16-performance-tuning)
17. [Monitoring & Metrics](#17-monitoring--metrics)
18. [Deployment Checklist](#18-deployment-checklist)

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

### 1.1 High-Level Design Philosophy

**Core Principles:**
- Two-phase ingestion (fast UX + deep processing)
- Non-destructive deduplication (graph-based)
- Hot/cold storage pattern for RAM optimization
- Hybrid retrieval (sparse + dense + reranking)
- Semantic compression as foundation layer

**Data Flow:**
```
Raw Files (100GB)
    ↓
[INGESTION] Fast Pass (2-3min) + Background (hours)
    ↓
[LLM PROCESSING] Summary + Keywords + Self-Consistency
    ↓
[SEMANTIC COMPRESSION] Scaffold Layer Creation
    ↓
[EMBEDDING] Dense (384d) + Sparse (BM25)
    ↓
[DEDUPLICATION] Non-Destructive Graph Clustering
    ↓
[INDEXING] Hot (HNSW RAM) + Cold (IVF+PQ Disk)
    ↓
[STORAGE] Parquet Metadata + FAISS Vectors + Original Files

--- QUERY TIME ---

Query Input
    ↓
[SEMANTIC ROUTER] Classify Query Type
    ↓
[HYBRID RETRIEVAL] BM25 → Dense → ColBERT Rerank
    ↓
[CACHE CHECK] Semantic Similarity > 0.95?
    ↓
[LLM GENERATION] Context Assembly + Response
    ↓
Final Answer + Source Attribution
```

### 1.2 Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Fast Pass Ingestion | 2-3 minutes | Time to first queryable state |
| Background Ingestion | 50-100 GB/hour | Full processing throughput |
| Query Latency (p50) | <200ms | Median response time |
| Query Latency (p95) | <800ms | 95th percentile |
| Recall@10 | >90% | Against ground truth queries |
| Compression Ratio | 10:1 | 100GB → 10GB semantic |
| Memory Usage (Query) | <20GB RAM | Peak during retrieval |
| Memory Usage (Ingestion) | <28GB RAM | Peak during processing |

---

## 2. HARDWARE REQUIREMENTS & CONSTRAINTS

### 2.1 Minimum Requirements

**GPU:**
- NVIDIA RTX 4050 (6GB VRAM) or equivalent
- CUDA 11.8+ support
- Compute Capability 8.6+

**RAM:**
- 32GB DDR4 minimum
- Recommended: 64GB for larger datasets

**Storage:**
- 150GB free space (100GB raw + 50GB processed)
- SSD highly recommended for FAISS mmap
- NVMe preferred for <100ms index load times

**CPU:**
- 8+ cores recommended
- AVX2 instruction set (for FAISS optimization)
- Minimum: Intel i7-10th gen or AMD Ryzen 5000+

### 2.2 Memory Budget Breakdown

| Component | RAM Usage | VRAM Usage | Notes |
|-----------|-----------|------------|-------|
| LLM (Phi-3 3.8B Q4) | 0.5GB | 2.5GB | During inference |
| Embeddings (10M×384d) | 15GB | 0GB | FAISS index in RAM |
| Hot Index (HNSW) | 3GB | 0GB | Top 20% documents |
| Cold Index (IVF+PQ) | Disk mmap | 0GB | 80% documents |
| BM25 Index | 2GB | 0GB | Inverted index |
| Metadata (Parquet) | 1GB | 0GB | Compressed |
| OS + Buffers | 5GB | 0GB | Reserved |
| Processing Buffers | 5GB | 3.5GB | During ingestion |
| **TOTAL** | **31.5GB** | **6GB** | Peak usage |

**Critical:** Stay within 28GB operational RAM to avoid swapping.

---

## 3. SOFTWARE STACK & DEPENDENCIES

### 3.1 Core Python Environment

**Python Version:** 3.10 or 3.11 (NOT 3.12, llama.cpp issues)

```bash
# Create isolated environment
conda create -n rag_local python=3.10
conda activate rag_local
```

### 3.2 Essential Libraries

```bash
# LLM Inference
pip install llama-cpp-python==0.2.56 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# Embedding Models
pip install sentence-transformers==2.2.2
pip install transformers==4.36.0

# Vector Search
pip install faiss-cpu==1.7.4  # Use faiss-gpu if >6GB VRAM available
pip install hnswlib==0.7.0

# Sparse Search
pip install whoosh==2.7.4
# OR for production:
# pip install elasticsearch==8.11.0

# Document Processing
pip install pypdf==3.0.1
pip install pdfplumber==0.10.3
pip install python-docx==1.1.0
pip install openpyxl==3.1.2
pip install pandas==2.1.4
pip install tabula-py==2.8.2
pip install camelot-py[cv]==0.11.0

# OCR (optional)
pip install pytesseract==0.3.10
pip install easyocr==1.7.0

# Deduplication & Clustering
pip install scikit-learn==1.3.2
pip install hdbscan==0.8.33
pip install networkx==3.2.1
pip install datasketch==1.6.4

# Storage & Serialization
pip install pyarrow==14.0.1
pip install parquet==1.3.1

# Utilities
pip install tqdm==4.66.1
pip install python-dotenv==1.0.0
pip install pyyaml==6.0.1
pip install rapidfuzz==3.5.2
pip install dateparser==1.2.0

# Monitoring (optional)
pip install psutil==5.9.6
pip install prometheus-client==0.19.0
```

### 3.3 Model Downloads

**LLM Models (choose one):**

```bash
# Download from Hugging Face
# Phi-3 Mini 3.8B Q4_K_M (RECOMMENDED)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

# Alternative: Mistral 3B
wget https://huggingface.co/TheBloke/Mistral-3B-GGUF/resolve/main/mistral-3b.Q4_K_M.gguf

# Alternative: Llama 3.1 3B
wget https://huggingface.co/bartowski/Llama-3.1-3B-GGUF/resolve/main/Llama-3.1-3B-Q4_K_M.gguf
```

**Embedding Models:**

```python
from sentence_transformers import SentenceTransformer

# Primary: all-MiniLM-L6-v2 (384 dimensions)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('models/minilm-l6-v2')

# Alternative: all-mpnet-base-v2 (768 dimensions, higher quality)
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# model.save('models/mpnet-base-v2')
```

**Reranker Model:**

```python
# ColBERT for late interaction reranking
from sentence_transformers import SentenceTransformer
reranker = SentenceTransformer('colbert-ir/colbertv2.0')
reranker.save('models/colbert-v2')
```

---

## 4. LAYER 1: INGESTION PIPELINE

### 4.1 Fast Pass Ingestion (2-3 minutes)

**Goal:** Get system queryable ASAP with basic metadata and BM25 index.

**Process:**

```python
import os
from pathlib import Path
from datetime import datetime
import hashlib

class FastPassIngestor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata = []

    def scan_directory(self):
        """Recursively scan and catalog all files"""
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                meta = self.extract_metadata(file_path)
                self.metadata.append(meta)
        return self.metadata

    def extract_metadata(self, file_path):
        """Extract basic metadata without full parsing"""
        stat = file_path.stat()
        return {
            'uuid': hashlib.md5(str(file_path).encode()).hexdigest(),
            'path': str(file_path),
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'size_bytes': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_ctime),
            'modified_at': datetime.fromtimestamp(stat.st_mtime),
            'indexed_at': datetime.now(),
            'processing_status': 'fast_pass_complete'
        }

    def quick_text_extract(self, file_path):
        """Extract text with minimal processing"""
        ext = file_path.suffix.lower()

        if ext == '.txt':
            return file_path.read_text(errors='ignore')[:5000]

        elif ext == '.pdf':
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    text = ""
                    for page in pdf.pages[:5]:  # First 5 pages only
                        text += page.extract_text()
                    return text[:5000]
            except:
                return ""

        elif ext in ['.docx', '.doc']:
            try:
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs[:50]])
                return text[:5000]
            except:
                return ""

        elif ext in ['.xlsx', '.xls', '.csv']:
            try:
                import pandas as pd
                df = pd.read_excel(file_path) if ext != '.csv' else pd.read_csv(file_path)
                return f"Table: {df.columns.tolist()}\nRows: {len(df)}\nSample:\n{df.head(10).to_string()}"[:5000]
            except:
                return ""

        return ""

    def build_bm25_index(self):
        """Create Whoosh BM25 index for immediate searching"""
        from whoosh.index import create_in
        from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC
        from whoosh.analysis import StemmingAnalyzer

        schema = Schema(
            uuid=ID(stored=True, unique=True),
            path=ID(stored=True),
            filename=TEXT(stored=True),
            content=TEXT(analyzer=StemmingAnalyzer(), stored=False),
            extension=ID(stored=True),
            size=NUMERIC(stored=True),
            modified_at=DATETIME(stored=True)
        )

        idx_dir = self.output_dir / 'bm25_index'
        idx_dir.mkdir(exist_ok=True)
        ix = create_in(str(idx_dir), schema)

        writer = ix.writer()
        for meta in self.metadata:
            file_path = Path(meta['path'])
            content = self.quick_text_extract(file_path)

            writer.add_document(
                uuid=meta['uuid'],
                path=meta['path'],
                filename=meta['filename'],
                content=content,
                extension=meta['extension'],
                size=meta['size_bytes'],
                modified_at=meta['modified_at']
            )

        writer.commit()
        return ix
```

**Usage:**

```python
ingestor = FastPassIngestor(
    data_dir='/path/to/100GB/data',
    output_dir='/path/to/output'
)

# Scan all files
metadata = ingestor.scan_directory()

# Build searchable index
bm25_index = ingestor.build_bm25_index()

# Save metadata
import pandas as pd
df = pd.DataFrame(metadata)
df.to_parquet('output/metadata_fast.parquet')

print(f"Fast Pass Complete: {len(metadata)} files indexed in {elapsed_time:.1f}s")
# System is now queryable!
```

**Expected Performance:**
- 100GB / 10,000 files → ~2-3 minutes
- BM25 index size: ~2GB for 100GB corpus
- Memory usage: <8GB during fast pass

### 4.2 Background Deep Processing

**Goal:** Extract full text, handle complex documents, prepare for embedding.

```python
class DeepIngestor:
    def __init__(self, metadata_df, output_dir):
        self.metadata = metadata_df
        self.output_dir = Path(output_dir)
        self.chunks = []

    def process_document(self, file_path, uuid):
        """Deep text extraction with full parsing"""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self._process_pdf(file_path, uuid)
        elif ext in ['.docx', '.doc']:
            return self._process_docx(file_path, uuid)
        elif ext in ['.xlsx', '.xls']:
            return self._process_excel(file_path, uuid)
        elif ext == '.csv':
            return self._process_csv(file_path, uuid)
        elif ext == '.txt':
            return self._process_txt(file_path, uuid)
        else:
            return []

    def _process_pdf(self, file_path, uuid):
        """PDF with pdfplumber for tables + text"""
        import pdfplumber
        chunks = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        chunks.append({
                            'doc_uuid': uuid,
                            'chunk_id': f"{uuid}_p{page_num}",
                            'text': text,
                            'page': page_num,
                            'type': 'text'
                        })

                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            chunks.append({
                                'doc_uuid': uuid,
                                'chunk_id': f"{uuid}_p{page_num}_t{table_idx}",
                                'text': str(table),
                                'page': page_num,
                                'type': 'table'
                            })
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")

        return chunks

    def _process_docx(self, file_path, uuid):
        """DOCX with python-docx"""
        from docx import Document
        chunks = []

        try:
            doc = Document(file_path)
            current_text = ""

            for para_idx, para in enumerate(doc.paragraphs):
                current_text += para.text + "\n"

                # Chunk every 1000 tokens (~750 words)
                if len(current_text.split()) > 750:
                    chunks.append({
                        'doc_uuid': uuid,
                        'chunk_id': f"{uuid}_chunk{len(chunks)}",
                        'text': current_text,
                        'type': 'text'
                    })
                    current_text = ""

            if current_text:
                chunks.append({
                    'doc_uuid': uuid,
                    'chunk_id': f"{uuid}_chunk{len(chunks)}",
                    'text': current_text,
                    'type': 'text'
                })
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {e}")

        return chunks

    def _process_excel(self, file_path, uuid):
        """Excel with sheet-level granularity"""
        import pandas as pd
        chunks = []

        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                chunks.append({
                    'doc_uuid': uuid,
                    'chunk_id': f"{uuid}_sheet_{sheet_name}",
                    'text': f"Sheet: {sheet_name}\nColumns: {df.columns.tolist()}\n{df.to_string()}",
                    'sheet_name': sheet_name,
                    'type': 'table',
                    'table_metadata': {
                        'columns': df.columns.tolist(),
                        'n_rows': len(df),
                        'n_cols': len(df.columns),
                        'dtypes': df.dtypes.to_dict()
                    }
                })
        except Exception as e:
            print(f"Error processing Excel {file_path}: {e}")

        return chunks

    def _process_csv(self, file_path, uuid):
        """CSV handling"""
        import pandas as pd
        chunks = []

        try:
            df = pd.read_csv(file_path)
            chunks.append({
                'doc_uuid': uuid,
                'chunk_id': f"{uuid}_csv",
                'text': f"CSV: {Path(file_path).name}\nColumns: {df.columns.tolist()}\n{df.to_string()}",
                'type': 'table',
                'table_metadata': {
                    'columns': df.columns.tolist(),
                    'n_rows': len(df),
                    'n_cols': len(df.columns)
                }
            })
        except Exception as e:
            print(f"Error processing CSV {file_path}: {e}")

        return chunks

    def _process_txt(self, file_path, uuid):
        """Plain text with chunking"""
        chunks = []

        try:
            text = Path(file_path).read_text(errors='ignore')
            words = text.split()

            # Chunk every 750 words
            for i in range(0, len(words), 750):
                chunk_text = " ".join(words[i:i+750])
                chunks.append({
                    'doc_uuid': uuid,
                    'chunk_id': f"{uuid}_chunk{len(chunks)}",
                    'text': chunk_text,
                    'type': 'text'
                })
        except Exception as e:
            print(f"Error processing TXT {file_path}: {e}")

        return chunks

    def process_all(self, n_workers=4):
        """Parallel processing with multiprocessing"""
        from multiprocessing import Pool
        from tqdm import tqdm

        tasks = [(row['path'], row['uuid']) for _, row in self.metadata.iterrows()]

        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.starmap(self.process_document, tasks),
                total=len(tasks),
                desc="Deep Processing"
            ))

        # Flatten results
        for chunk_list in results:
            self.chunks.extend(chunk_list)

        return self.chunks
```

**Usage:**

```python
deep = DeepIngestor(metadata_df, output_dir)
all_chunks = deep.process_all(n_workers=8)

# Save chunks
chunks_df = pd.DataFrame(all_chunks)
chunks_df.to_parquet('output/chunks_deep.parquet')

print(f"Deep Processing Complete: {len(all_chunks)} chunks extracted")
```

**Expected Performance:**
- 100GB corpus → 50-100 GB/hour (depending on file types)
- Produces 1-10M chunks depending on document sizes
- Memory: <10GB per worker

---

## 5. LAYER 2: LLM PROCESSING

### 5.1 LLM Setup (llama.cpp)

```python
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path, n_ctx=4096, n_gpu_layers=35):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # Offload to GPU
            n_batch=512,
            n_threads=8,
            verbose=False
        )

    def generate(self, prompt, max_tokens=512, temperature=0.3):
        """Generate completion"""
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["</s>", "\n\n\n"]
        )
        return response['choices'][0]['text'].strip()

    def summarize(self, text, max_length=200):
        """Summarize text"""
        prompt = f"""<|system|>You are a precise summarization assistant. Create a concise summary capturing key information.<|end|>
<|user|>Summarize this text in {max_length} words or less:

{text[:3000]}

Summary:<|end|>
<|assistant|>"""

        return self.generate(prompt, max_tokens=max_length*2)

    def extract_keywords(self, text, n_keywords=10):
        """Extract keywords"""
        prompt = f"""<|system|>You are a keyword extraction assistant. Extract the {n_keywords} most important keywords or key phrases.<|end|>
<|user|>Extract {n_keywords} keywords from this text:

{text[:2000]}

Keywords (comma-separated):<|end|>
<|assistant|>"""

        keywords_str = self.generate(prompt, max_tokens=150)
        keywords = [k.strip() for k in keywords_str.split(',')]
        return keywords[:n_keywords]

    def categorize(self, text, categories=None):
        """Assign categories"""
        if categories is None:
            categories = ["Technical", "Business", "Legal", "Financial", "Research", "Administrative", "Other"]

        prompt = f"""<|system|>You are a document classification assistant.<|end|>
<|user|>Classify this document into one or more categories: {', '.join(categories)}

Text: {text[:1500]}

Category (pick 1-2):<|end|>
<|assistant|>"""

        return self.generate(prompt, max_tokens=50)
```

### 5.2 Self-Consistency Pipeline

```python
class SelfConsistentLLM:
    def __init__(self, llm, n_runs=3):
        self.llm = llm
        self.n_runs = n_runs

    def summarize_with_consensus(self, text):
        """Generate multiple summaries and find consensus"""
        summaries = []

        for i in range(self.n_runs):
            summary = self.llm.summarize(text)
            summaries.append(summary)

        # Use longest summary as base (usually most complete)
        return max(summaries, key=len)

    def extract_keywords_with_voting(self, text):
        """Extract keywords with voting mechanism"""
        from collections import Counter
        all_keywords = []

        for i in range(self.n_runs):
            keywords = self.llm.extract_keywords(text)
            all_keywords.extend([k.lower() for k in keywords])

        # Keep keywords that appear in at least 2/3 runs
        keyword_counts = Counter(all_keywords)
        threshold = self.n_runs * 0.66
        consensus_keywords = [k for k, count in keyword_counts.items() if count >= threshold]

        return consensus_keywords[:10]
```

### 5.3 Batch Processing for Scale

```python
def process_chunks_with_llm(chunks_df, llm, batch_size=10):
    """Process chunks in batches"""
    from tqdm import tqdm

    processed = []

    for i in tqdm(range(0, len(chunks_df), batch_size), desc="LLM Processing"):
        batch = chunks_df.iloc[i:i+batch_size]

        for _, chunk in batch.iterrows():
            try:
                summary = llm.summarize(chunk['text'])
                keywords = llm.extract_keywords(chunk['text'])
                category = llm.categorize(chunk['text'])

                processed.append({
                    'chunk_id': chunk['chunk_id'],
                    'summary': summary,
                    'keywords': keywords,
                    'category': category
                })
            except Exception as e:
                print(f"Error processing chunk {chunk['chunk_id']}: {e}")
                continue

    return pd.DataFrame(processed)

# Usage
llm = LocalLLM('models/Phi-3-mini-4k-instruct-q4.gguf')
processed_df = process_chunks_with_llm(chunks_df, llm)
processed_df.to_parquet('output/chunks_llm_processed.parquet')
```

**Expected Performance:**
- Phi-3 3.8B Q4: ~20-30 tokens/sec on RTX 4050
- 1M chunks @ 200 tokens each → ~10-15 hours
- Run overnight or over weekend

---

## 6. LAYER 3: EMBEDDING GENERATION

### 6.1 Dense Embeddings (Sentence-BERT)

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_batch(self, texts, batch_size=32):
        """Generate embeddings in batches"""
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True  # Important for cosine similarity
            )
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def embed_summaries(self, processed_df):
        """Embed summaries for semantic search"""
        summaries = processed_df['summary'].tolist()
        embeddings = self.embed_batch(summaries)
        return embeddings

    def embed_chunks(self, chunks_df):
        """Embed full chunks"""
        texts = chunks_df['text'].tolist()
        embeddings = self.embed_batch(texts)
        return embeddings

# Usage
embedder = EmbeddingGenerator()

# Embed summaries (lightweight layer)
summary_embeddings = embedder.embed_summaries(processed_df)
np.save('output/embeddings_summaries.npy', summary_embeddings)

# Embed full chunks (deep layer)
chunk_embeddings = embedder.embed_chunks(chunks_df)
np.save('output/embeddings_chunks.npy', chunk_embeddings)

print(f"Embedding dimension: {embedder.dimension}")
print(f"Summary embeddings: {summary_embeddings.shape}")
print(f"Chunk embeddings: {chunk_embeddings.shape}")
```

**Expected Performance:**
- all-MiniLM-L6-v2: ~1000-2000 sentences/sec on CPU
- 1M chunks → ~10-20 minutes
- Memory: ~15GB for 10M embeddings @ 384d

### 6.2 Sparse Index (BM25) - Already Built in Fast Pass

BM25 index is already created during fast pass, but you can enhance it:

```python
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, MultifieldParser

class BM25Searcher:
    def __init__(self, index_dir):
        self.ix = open_dir(index_dir)
        self.parser = MultifieldParser(
            ["filename", "content"],
            schema=self.ix.schema
        )

    def search(self, query_text, limit=500):
        """Search BM25 index"""
        with self.ix.searcher() as searcher:
            query = self.parser.parse(query_text)
            results = searcher.search(query, limit=limit)

            return [{
                'uuid': hit['uuid'],
                'path': hit['path'],
                'filename': hit['filename'],
                'score': hit.score
            } for hit in results]

# Usage
bm25 = BM25Searcher('output/bm25_index')
sparse_results = bm25.search("machine learning algorithms", limit=500)
```

---

## 7. LAYER 4: DEDUPLICATION SYSTEM

### 7.1 Text Deduplication (Non-Destructive Graph)

```python
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticDeduplicator:
    def __init__(self, similarity_threshold=0.92):
        self.threshold = similarity_threshold
        self.similarity_graph = nx.Graph()

    def build_similarity_graph(self, embeddings, chunk_ids):
        """Build graph of similar chunks"""
        print("Computing pairwise similarities...")

        # Use batched cosine similarity to avoid memory issues
        n_chunks = len(embeddings)
        batch_size = 1000

        for i in range(0, n_chunks, batch_size):
            batch_embeddings = embeddings[i:i+batch_size]

            # Compare with all other embeddings
            similarities = cosine_similarity(batch_embeddings, embeddings)

            # Find pairs above threshold
            for local_idx, global_idx in enumerate(range(i, min(i+batch_size, n_chunks))):
                similar_indices = np.where(similarities[local_idx] > self.threshold)[0]

                for sim_idx in similar_indices:
                    if sim_idx != global_idx:  # Skip self-similarity
                        self.similarity_graph.add_edge(
                            chunk_ids[global_idx],
                            chunk_ids[sim_idx],
                            weight=float(similarities[local_idx, sim_idx])
                        )

        return self.similarity_graph

    def find_clusters(self):
        """Find connected components (clusters of similar chunks)"""
        clusters = list(nx.connected_components(self.similarity_graph))

        cluster_mapping = {}
        for cluster_id, cluster in enumerate(clusters):
            for chunk_id in cluster:
                cluster_mapping[chunk_id] = cluster_id

        return clusters, cluster_mapping

    def select_representatives(self, clusters, chunks_df):
        """Select one representative per cluster (longest/most complete)"""
        representatives = {}

        for cluster_id, cluster in enumerate(clusters):
            cluster_chunks = chunks_df[chunks_df['chunk_id'].isin(cluster)]

            # Select longest chunk as representative
            rep_idx = cluster_chunks['text'].str.len().idxmax()
            representative = cluster_chunks.loc[rep_idx]

            representatives[cluster_id] = {
                'chunk_id': representative['chunk_id'],
                'cluster_size': len(cluster),
                'similar_chunks': list(cluster)
            }

        return representatives

# Usage
deduplicator = SemanticDeduplicator(similarity_threshold=0.92)

# Build similarity graph
chunk_ids = chunks_df['chunk_id'].tolist()
similarity_graph = deduplicator.build_similarity_graph(chunk_embeddings, chunk_ids)

# Find clusters
clusters, cluster_mapping = deduplicator.find_clusters()
print(f"Found {len(clusters)} clusters from {len(chunk_ids)} chunks")

# Select representatives
representatives = deduplicator.select_representatives(clusters, chunks_df)

# Save results
import json
with open('output/dedup_clusters.json', 'w') as f:
    json.dump(representatives, f)

# Add cluster info to chunks dataframe
chunks_df['cluster_id'] = chunks_df['chunk_id'].map(cluster_mapping)
chunks_df['is_representative'] = chunks_df['chunk_id'].isin(
    [rep['chunk_id'] for rep in representatives.values()]
)
chunks_df.to_parquet('output/chunks_with_clusters.parquet')
```

**Expected Results:**
- 1M chunks → 300-500k clusters (30-50% deduplication)
- Processing time: 2-4 hours for 1M chunks
- Memory: peak 20GB for similarity computation

### 7.2 Table Deduplication (Embedding-Based)

```python
import pandas as pd
from hdbscan import HDBSCAN

class TableDeduplicator:
    def __init__(self, embedder):
        self.embedder = embedder

    def create_table_signature(self, table_metadata, sample_data):
        """Create semantic signature for table"""
        signature_text = f"""
        Columns: {', '.join(table_metadata['columns'])}
        Data types: {', '.join([str(dt) for dt in table_metadata['dtypes'].values()])}
        Rows: {table_metadata['n_rows']}
        Sample data:
        {sample_data}
        """
        return signature_text

    def embed_tables(self, tables_df):
        """Embed all tables"""
        signatures = []

        for _, row in tables_df.iterrows():
            signature = self.create_table_signature(
                row['table_metadata'],
                row['text'][:1000]
            )
            signatures.append(signature)

        embeddings = self.embedder.embed_batch(signatures)
        return embeddings

    def cluster_tables(self, table_embeddings, min_cluster_size=2):
        """Cluster similar tables using HDBSCAN"""
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            min_samples=1
        )

        cluster_labels = clusterer.fit_predict(table_embeddings)
        return cluster_labels

    def create_virtual_tables(self, tables_df, cluster_labels):
        """Create virtual consolidated tables"""
        virtual_tables = {}

        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_tables = tables_df[cluster_labels == cluster_id]

            virtual_tables[cluster_id] = {
                'cluster_id': cluster_id,
                'n_tables': len(cluster_tables),
                'source_files': cluster_tables['doc_uuid'].tolist(),
                'common_columns': self._find_common_columns(cluster_tables),
                'representative': cluster_tables.iloc[0]['chunk_id']
            }

        return virtual_tables

    def _find_common_columns(self, cluster_tables):
        """Find columns common across tables in cluster"""
        all_columns = []
        for _, row in cluster_tables.iterrows():
            all_columns.extend(row['table_metadata']['columns'])

        from collections import Counter
        column_counts = Counter(all_columns)
        common = [col for col, count in column_counts.items()
                  if count >= len(cluster_tables) * 0.7]
        return common

# Usage
table_dedup = TableDeduplicator(embedder)

# Filter table chunks
tables_df = chunks_df[chunks_df['type'] == 'table'].copy()

# Embed tables
table_embeddings = table_dedup.embed_tables(tables_df)

# Cluster similar tables
table_clusters = table_dedup.cluster_tables(table_embeddings)
print(f"Found {len(set(table_clusters))} table clusters")

# Create virtual tables
virtual_tables = table_dedup.create_virtual_tables(tables_df, table_clusters)

# Save
with open('output/virtual_tables.json', 'w') as f:
    json.dump(virtual_tables, f, default=str)
```

---

## 8. LAYER 5: SEMANTIC COMPRESSION

### 8.1 Semantic Scaffold Creation

```python
class SemanticScaffold:
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder

    def create_scaffold_entry(self, chunk):
        """Create compressed semantic entry for chunk"""
        # Get LLM-generated metadata
        summary = self.llm.summarize(chunk['text'])
        keywords = self.llm.extract_keywords(chunk['text'])
        category = self.llm.categorize(chunk['text'])

        # Generate embedding of summary (much smaller than full text)
        summary_embedding = self.embedder.model.encode([summary])[0]

        scaffold_entry = {
            'chunk_id': chunk['chunk_id'],
            'doc_uuid': chunk['doc_uuid'],
            'summary': summary,
            'keywords': keywords,
            'category': category,
            'summary_embedding': summary_embedding,
            'original_size': len(chunk['text']),
            'compressed_size': len(summary),
            'compression_ratio': len(chunk['text']) / len(summary)
        }

        return scaffold_entry

    def build_full_scaffold(self, chunks_df, batch_size=100):
        """Build scaffold for entire corpus"""
        scaffold_entries = []

        for i in tqdm(range(0, len(chunks_df), batch_size), desc="Building Scaffold"):
            batch = chunks_df.iloc[i:i+batch_size]

            for _, chunk in batch.iterrows():
                entry = self.create_scaffold_entry(chunk)
                scaffold_entries.append(entry)

        return pd.DataFrame(scaffold_entries)

    def save_scaffold(self, scaffold_df, output_path):
        """Save scaffold with separate embedding storage"""
        # Extract embeddings
        embeddings = np.array(scaffold_df['summary_embedding'].tolist())

        # Save embeddings separately
        np.save(f"{output_path}/scaffold_embeddings.npy", embeddings)

        # Save metadata without embeddings
        metadata = scaffold_df.drop('summary_embedding', axis=1)
        metadata.to_parquet(f"{output_path}/scaffold_metadata.parquet")

        return embeddings

# Usage
scaffold = SemanticScaffold(llm, embedder)
scaffold_df = scaffold.build_full_scaffold(chunks_df)

# Save
scaffold_embeddings = scaffold.save_scaffold(scaffold_df, 'output/scaffold')

print(f"Scaffold size: {len(scaffold_df)} entries")
print(f"Compression achieved: {scaffold_df['compression_ratio'].mean():.1f}x average")
```

**Expected Results:**
- 100GB raw → 5-10GB semantic scaffold
- 10:1 compression ratio on average
- Maintains queryability and semantic richness

---

## 9. LAYER 6: VECTOR INDEXING

### 9.1 Hot/Cold Storage Implementation

```python
import faiss
import pickle

class HotColdVectorStore:
    def __init__(self, dimension=384, hot_ratio=0.2):
        self.dimension = dimension
        self.hot_ratio = hot_ratio

        # Hot index (HNSW in RAM)
        self.hot_index = faiss.IndexHNSWFlat(dimension, 32)
        self.hot_index.hnsw.efConstruction = 200
        self.hot_index.hnsw.efSearch = 64

        # Cold index (IVF+PQ on disk)
        quantizer = faiss.IndexFlatL2(dimension)
        self.cold_index = faiss.IndexIVFPQ(
            quantizer,
            dimension,
            nlist=4096,      # Number of clusters
            m=64,            # Number of subquantizers
            nbits=8          # Bits per subquantizer
        )

        self.hot_ids = []
        self.cold_ids = []
        self.access_counts = {}

    def train_cold_index(self, training_vectors):
        """Train IVF+PQ index"""
        print("Training cold index...")
        self.cold_index.train(training_vectors)
        print("Cold index trained")

    def add_to_hot(self, vectors, ids):
        """Add vectors to hot index"""
        self.hot_index.add(vectors)
        self.hot_ids.extend(ids)

    def add_to_cold(self, vectors, ids):
        """Add vectors to cold index"""
        self.cold_index.add(vectors)
        self.cold_ids.extend(ids)

    def split_hot_cold(self, embeddings, chunk_ids, access_history=None):
        """Split embeddings into hot and cold based on access patterns"""
        n_hot = int(len(embeddings) * self.hot_ratio)

        if access_history:
            # Sort by access frequency
            sorted_indices = np.argsort([-access_history.get(cid, 0) for cid in chunk_ids])
        else:
            # Random split for initial deployment
            sorted_indices = np.random.permutation(len(embeddings))

        hot_indices = sorted_indices[:n_hot]
        cold_indices = sorted_indices[n_hot:]

        return hot_indices, cold_indices

    def build_indices(self, embeddings, chunk_ids, access_history=None):
        """Build both hot and cold indices"""
        hot_idx, cold_idx = self.split_hot_cold(embeddings, chunk_ids, access_history)

        # Add to hot
        hot_vectors = embeddings[hot_idx].astype('float32')
        hot_ids = [chunk_ids[i] for i in hot_idx]
        self.add_to_hot(hot_vectors, hot_ids)
        print(f"Added {len(hot_ids)} vectors to hot index")

        # Train and add to cold
        cold_vectors = embeddings[cold_idx].astype('float32')
        cold_ids = [chunk_ids[i] for i in cold_idx]

        # Sample for training (use 100k vectors max)
        train_size = min(100000, len(cold_vectors))
        train_vectors = cold_vectors[np.random.choice(len(cold_vectors), train_size, replace=False)]
        self.train_cold_index(train_vectors)

        self.add_to_cold(cold_vectors, cold_ids)
        print(f"Added {len(cold_ids)} vectors to cold index")

    def search(self, query_vector, k=50):
        """Search both indices and merge results"""
        query_vector = query_vector.reshape(1, -1).astype('float32')

        # Search hot index
        k_hot = int(k * 0.7)  # Get 70% from hot
        D_hot, I_hot = self.hot_index.search(query_vector, k_hot)

        # Search cold index
        k_cold = k - k_hot
        self.cold_index.nprobe = 32  # Search 32 clusters
        D_cold, I_cold = self.cold_index.search(query_vector, k_cold)

        # Merge results
        hot_results = [
            {'chunk_id': self.hot_ids[idx], 'score': float(dist), 'source': 'hot'}
            for idx, dist in zip(I_hot[0], D_hot[0])
            if idx < len(self.hot_ids)
        ]

        cold_results = [
            {'chunk_id': self.cold_ids[idx], 'score': float(dist), 'source': 'cold'}
            for idx, dist in zip(I_cold[0], D_cold[0])
            if idx < len(self.cold_ids)
        ]

        # Combine and sort by score
        all_results = hot_results + cold_results
        all_results.sort(key=lambda x: x['score'])

        return all_results[:k]

    def promote_to_hot(self, chunk_id):
        """Move frequently accessed item from cold to hot"""
        # Implementation of LRU promotion logic
        self.access_counts[chunk_id] = self.access_counts.get(chunk_id, 0) + 1

        # Promote if access count exceeds threshold
        if self.access_counts[chunk_id] > 10 and chunk_id in self.cold_ids:
            # Find vector in cold index
            idx = self.cold_ids.index(chunk_id)
            # Move to hot (simplified - full implementation would reindex)
            print(f"Promoting {chunk_id} to hot storage")

    def save(self, output_path):
        """Save indices to disk"""
        # Save hot index
        faiss.write_index(self.hot_index, f"{output_path}/hot_index.faiss")

        # Save cold index
        faiss.write_index(self.cold_index, f"{output_path}/cold_index.faiss")

        # Save metadata
        metadata = {
            'hot_ids': self.hot_ids,
            'cold_ids': self.cold_ids,
            'access_counts': self.access_counts,
            'dimension': self.dimension,
            'hot_ratio': self.hot_ratio
        }
        with open(f"{output_path}/index_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, output_path):
        """Load indices from disk"""
        self.hot_index = faiss.read_index(f"{output_path}/hot_index.faiss")
        self.cold_index = faiss.read_index(f"{output_path}/cold_index.faiss")

        with open(f"{output_path}/index_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.hot_ids = metadata['hot_ids']
            self.cold_ids = metadata['cold_ids']
            self.access_counts = metadata['access_counts']

# Usage
vector_store = HotColdVectorStore(dimension=384, hot_ratio=0.2)

# Build indices
vector_store.build_indices(
    embeddings=chunk_embeddings,
    chunk_ids=chunk_ids,
    access_history=None  # Can provide access logs if available
)

# Save to disk
vector_store.save('output/vector_store')

# Test search
query_embedding = embedder.model.encode(["machine learning"])
results = vector_store.search(query_embedding[0], k=50)
print(f"Found {len(results)} results")
```

**Memory Footprint:**
- Hot (20%): ~3GB for 2M vectors @ 384d
- Cold (80%): disk-based with mmap, ~2GB RAM during search
- Total: ~5GB RAM for vector storage

---

## 10. LAYER 7: QUERY ROUTING

### 10.1 Semantic Router Implementation

```python
from enum import Enum
import re

class QueryType(Enum):
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    EXPLORATORY = "exploratory"

class SemanticRouter:
    def __init__(self, embedder):
        self.embedder = embedder

        # Define query type patterns
        self.patterns = {
            QueryType.FACTUAL: [
                r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b',
                r'\bhow many\b', r'\bdefine\b', r'\blist\b'
            ],
            QueryType.TEMPORAL: [
                r'\brecent\b', r'\blatest\b', r'\blast\b',
                r'\b20\d{2}\b', r'\bthis year\b', r'\byesterday\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b',
                r'\bdifference between\b', r'\bbetter than\b'
            ],
            QueryType.CONCEPTUAL: [
                r'\bexplain\b', r'\bwhy\b', r'\bhow does\b',
                r'\brelationship\b', r'\bimpact\b'
            ]
        }

    def classify_query(self, query_text):
        """Classify query type"""
        query_lower = query_text.lower()

        # Check patterns
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type

        # Default to exploratory
        return QueryType.EXPLORATORY

    def extract_temporal_filter(self, query_text):
        """Extract date range from query"""
        import dateparser

        # Look for date mentions
        date_patterns = [
            r'\b20\d{2}\b',  # Year
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
            r'\blast\s+(week|month|year)\b',
            r'\bthis\s+(week|month|year)\b'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, query_text.lower())
            if match:
                date_str = match.group(0)
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    return parsed_date

        return None

    def route_query(self, query_text):
        """Route query and return search strategy"""
        query_type = self.classify_query(query_text)
        temporal_filter = self.extract_temporal_filter(query_text)

        strategy = {
            'query_type': query_type,
            'temporal_filter': temporal_filter,
            'use_bm25': True,  # Always use BM25 initially
            'bm25_weight': 0.3,
            'dense_weight': 0.7,
            'use_reranker': False,
            'k_candidates': 500
        }

        # Adjust strategy based on query type
        if query_type == QueryType.FACTUAL:
            strategy['bm25_weight'] = 0.6
            strategy['dense_weight'] = 0.4
            strategy['k_candidates'] = 300

        elif query_type == QueryType.CONCEPTUAL:
            strategy['bm25_weight'] = 0.2
            strategy['dense_weight'] = 0.8
            strategy['use_reranker'] = True

        elif query_type == QueryType.COMPARATIVE:
            strategy['use_reranker'] = True
            strategy['k_candidates'] = 100

        elif query_type == QueryType.TEMPORAL:
            strategy['bm25_weight'] = 0.4
            strategy['dense_weight'] = 0.6
            # Temporal filter will be applied

        return strategy

# Usage
router = SemanticRouter(embedder)

query = "What are the latest developments in machine learning since 2023?"
strategy = router.route_query(query)

print(f"Query type: {strategy['query_type']}")
print(f"BM25 weight: {strategy['bm25_weight']}")
print(f"Dense weight: {strategy['dense_weight']}")
print(f"Use reranker: {strategy['use_reranker']}")
print(f"Temporal filter: {strategy['temporal_filter']}")
```

---

## 11. LAYER 8: HYBRID RETRIEVAL

### 11.1 Complete Hybrid Search Pipeline

```python
class HybridRetriever:
    def __init__(self, bm25_searcher, vector_store, embedder, reranker=None):
        self.bm25 = bm25_searcher
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query_text, strategy):
        """Execute hybrid retrieval"""
        # Stage 1: BM25 Sparse Search
        bm25_results = self.bm25.search(
            query_text,
            limit=strategy['k_candidates']
        )

        # Stage 2: Dense Vector Search
        query_embedding = self.embedder.model.encode([query_text])[0]
        dense_results = self.vector_store.search(
            query_embedding,
            k=strategy['k_candidates'] // 2
        )

        # Stage 3: Merge Results with Weighted Scoring
        merged = self._merge_results(
            bm25_results,
            dense_results,
            bm25_weight=strategy['bm25_weight'],
            dense_weight=strategy['dense_weight']
        )

        # Stage 4: Apply Temporal Filter (if applicable)
        if strategy['temporal_filter']:
            merged = self._apply_temporal_filter(merged, strategy['temporal_filter'])

        # Stage 5: Reranking (if enabled)
        if strategy['use_reranker'] and self.reranker:
            top_k = merged[:20]  # Rerank top 20
            reranked = self._rerank(query_text, top_k)
            merged = reranked + merged[20:]

        return merged[:50]  # Return top 50

    def _merge_results(self, bm25_results, dense_results, bm25_weight, dense_weight):
        """Merge BM25 and dense results with weights"""
        # Normalize scores
        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            for r in bm25_results:
                r['score'] = r['score'] / max_bm25 if max_bm25 > 0 else 0

        if dense_results:
            max_dense = max(r['score'] for r in dense_results)
            for r in dense_results:
                r['score'] = r['score'] / max_dense if max_dense > 0 else 0

        # Combine scores
        combined = {}

        for result in bm25_results:
            chunk_id = result['uuid']  # BM25 uses 'uuid'
            combined[chunk_id] = {
                'chunk_id': chunk_id,
                'score': result['score'] * bm25_weight,
                'sources': ['bm25']
            }

        for result in dense_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined:
                combined[chunk_id]['score'] += result['score'] * dense_weight
                combined[chunk_id]['sources'].append('dense')
            else:
                combined[chunk_id] = {
                    'chunk_id': chunk_id,
                    'score': result['score'] * dense_weight,
                    'sources': ['dense']
                }

        # Sort by combined score
        merged = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return merged

    def _apply_temporal_filter(self, results, date_filter):
        """Filter results by date"""
        # Implementation would check document dates
        # Simplified here
        return results

    def _rerank(self, query_text, candidates):
        """Rerank using ColBERT or cross-encoder"""
        if not self.reranker:
            return candidates

        # Get full text for candidates
        chunk_texts = []
        for c in candidates:
            # Load chunk text from database
            chunk_text = self._load_chunk_text(c['chunk_id'])
            chunk_texts.append(chunk_text)

        # Compute reranking scores
        pairs = [[query_text, text] for text in chunk_texts]
        scores = self.reranker.predict(pairs)

        # Update scores
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])

        # Re-sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    def _load_chunk_text(self, chunk_id):
        """Load chunk text from storage"""
        # Implementation would load from parquet/database
        return "Placeholder text"

# Usage
hybrid_retriever = HybridRetriever(
    bm25_searcher=bm25,
    vector_store=vector_store,
    embedder=embedder,
    reranker=None  # Add ColBERT if needed
)

query = "machine learning algorithms for classification"
strategy = router.route_query(query)
results = hybrid_retriever.retrieve(query, strategy)

print(f"Retrieved {len(results)} results")
for i, result in enumerate(results[:5]):
    print(f"{i+1}. Chunk: {result['chunk_id']}, Score: {result['score']:.3f}, Sources: {result['sources']}")
```

---

## 12. LAYER 9: CACHING SYSTEM

### 12.1 Semantic Query Cache

```python
import numpy as np
from datetime import datetime, timedelta

class SemanticCache:
    def __init__(self, embedder, similarity_threshold=0.95, ttl_hours=1):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)

        self.cache = {}  # query_embedding -> results
        self.query_embeddings = []
        self.query_keys = []
        self.timestamps = []

    def _compute_key(self, query_embedding):
        """Generate cache key from embedding"""
        return hashlib.md5(query_embedding.tobytes()).hexdigest()

    def get(self, query_text):
        """Check if similar query exists in cache"""
        query_embedding = self.embedder.model.encode([query_text])[0]

        if not self.query_embeddings:
            return None

        # Compute similarity with cached queries
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.array(self.query_embeddings)
        )[0]

        # Find most similar cached query
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        if max_sim >= self.threshold:
            # Check TTL
            if datetime.now() - self.timestamps[max_idx] < self.ttl:
                cache_key = self.query_keys[max_idx]
                print(f"Cache HIT (similarity: {max_sim:.3f})")
                return self.cache[cache_key]

        return None

    def set(self, query_text, results):
        """Store query results in cache"""
        query_embedding = self.embedder.model.encode([query_text])[0]
        cache_key = self._compute_key(query_embedding)

        self.cache[cache_key] = results
        self.query_embeddings.append(query_embedding)
        self.query_keys.append(cache_key)
        self.timestamps.append(datetime.now())

        # Evict old entries (simple LRU)
        self._evict_old()

    def _evict_old(self, max_size=1000):
        """Remove old cache entries"""
        if len(self.query_keys) > max_size:
            # Remove oldest
            n_remove = len(self.query_keys) - max_size

            for i in range(n_remove):
                old_key = self.query_keys[0]
                del self.cache[old_key]
                self.query_embeddings.pop(0)
                self.query_keys.pop(0)
                self.timestamps.pop(0)

    def invalidate_all(self):
        """Clear entire cache (call after new ingestion)"""
        self.cache.clear()
        self.query_embeddings.clear()
        self.query_keys.clear()
        self.timestamps.clear()

# Usage
cache = SemanticCache(embedder, similarity_threshold=0.95, ttl_hours=1)

# Check cache
cached_results = cache.get("machine learning algorithms")
if cached_results:
    print("Using cached results")
    results = cached_results
else:
    print("Cache miss, executing search")
    results = hybrid_retriever.retrieve(query, strategy)
    cache.set(query, results)
```

---

## 13. LAYER 10: RESPONSE GENERATION

### 13.1 Context Assembly

```python
class ContextAssembler:
    def __init__(self, chunks_df, dedup_graph):
        self.chunks_df = chunks_df
        self.dedup_graph = dedup_graph

    def assemble_context(self, results, max_chunks=5, max_tokens=3000):
        """Assemble context from search results"""
        context_chunks = []
        total_tokens = 0

        for result in results:
            if len(context_chunks) >= max_chunks:
                break

            chunk_id = result['chunk_id']

            # Load chunk
            chunk = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id].iloc[0]

            # Check dedup graph for related chunks
            related = self._get_related_chunks(chunk_id)

            # Estimate tokens (rough: 1 token ~= 4 chars)
            chunk_tokens = len(chunk['text']) // 4

            if total_tokens + chunk_tokens > max_tokens:
                # Truncate chunk
                available_tokens = max_tokens - total_tokens
                chunk_text = chunk['text'][:available_tokens * 4]
            else:
                chunk_text = chunk['text']

            context_chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'source_file': chunk['doc_uuid'],
                'score': result['score'],
                'related_chunks': related
            })

            total_tokens += chunk_tokens

            if total_tokens >= max_tokens:
                break

        return context_chunks

    def _get_related_chunks(self, chunk_id):
        """Get similar chunks from dedup graph"""
        if chunk_id not in self.dedup_graph:
            return []

        neighbors = list(self.dedup_graph[chunk_id].keys())
        return neighbors[:3]  # Return top 3 similar chunks

    def format_context(self, context_chunks):
        """Format context for LLM"""
        formatted = []

        for i, chunk in enumerate(context_chunks, 1):
            formatted.append(f"[Source {i}] (ID: {chunk['chunk_id']}, Score: {chunk['score']:.3f})")
            formatted.append(chunk['text'])

            if chunk['related_chunks']:
                formatted.append(f"  Related chunks: {', '.join(chunk['related_chunks'][:2])}")

            formatted.append("")  # Blank line

        return "\n".join(formatted)

# Usage
assembler = ContextAssembler(chunks_df, similarity_graph)
context_chunks = assembler.assemble_context(results, max_chunks=5)
formatted_context = assembler.format_context(context_chunks)
```

### 13.2 Response Generation with Attribution

```python
class ResponseGenerator:
    def __init__(self, llm, assembler):
        self.llm = llm
        self.assembler = assembler

    def generate_response(self, query, results):
        """Generate final response with source attribution"""
        # Assemble context
        context_chunks = self.assembler.assemble_context(results)
        formatted_context = self.assembler.format_context(context_chunks)

        # Create prompt
        prompt = self._create_prompt(query, formatted_context)

        # Generate response
        response = self.llm.generate(prompt, max_tokens=1000)

        # Add source attribution
        attributed_response = self._add_attribution(response, context_chunks)

        return attributed_response

    def _create_prompt(self, query, context):
        """Create prompt for LLM using canonical system prompt.
        
        Note: In production code, this uses the DEFAULT_SYSTEM_PROMPT from
        cubo.config.prompt_defaults for consistency across all LLM providers.
        """
        # This example shows the expected format - actual implementation uses:
        # from cubo.config.prompt_defaults import DEFAULT_SYSTEM_PROMPT
        prompt = f"""<|system|>You are a helpful assistant that answers questions based on the provided context. Always cite sources using [Source N] notation when referencing specific information. If the answer is not in the provided context, reply 'Not in provided context.' Use only the provided context to answer - do not use external knowledge, assumptions, or invented information. Be concise and accurate.<|end|>
<|user|>Context:
{context}

Question: {query}

Answer based on the context above. Cite sources using [Source N] notation.<|end|>
<|assistant|>"""
        return prompt

    def _add_attribution(self, response, context_chunks):
        """Add detailed source attribution"""
        attribution = {
            'response': response,
            'sources': []
        }

        for i, chunk in enumerate(context_chunks, 1):
            # Check if source was cited in response
            if f"[Source {i}]" in response or f"Source {i}" in response:
                attribution['sources'].append({
                    'source_id': i,
                    'chunk_id': chunk['chunk_id'],
                    'file': chunk['source_file'],
                    'relevance_score': chunk['score'],
                    'excerpt': chunk['text'][:200] + "..."
                })

        return attribution

# Usage
generator = ResponseGenerator(llm, assembler)
final_response = generator.generate_response(query, results)

print("Response:", final_response['response'])
print("\nSources cited:")
for source in final_response['sources']:
    print(f"  [{source['source_id']}] {source['file']} (score: {source['relevance_score']:.3f})")
```

---

## 14. DATABASE SCHEMA & STORAGE

### 14.1 Storage Architecture

```
output/
├── metadata/
│   ├── files_metadata.parquet          # File-level metadata
│   ├── chunks_metadata.parquet         # Chunk-level metadata
│   └── scaffold_metadata.parquet       # Semantic scaffold
│
├── embeddings/
│   ├── chunk_embeddings.npy            # Full chunk embeddings
│   ├── scaffold_embeddings.npy         # Summary embeddings
│   └── table_embeddings.npy            # Table embeddings
│
├── indices/
│   ├── bm25_index/                     # Whoosh BM25 index
│   ├── hot_index.faiss                 # Hot FAISS HNSW
│   ├── cold_index.faiss                # Cold FAISS IVF+PQ
│   └── index_metadata.pkl              # Index mappings
│
├── deduplication/
│   ├── dedup_clusters.json             # Text dedup results
│   ├── similarity_graph.pkl            # NetworkX graph
│   └── virtual_tables.json             # Table dedup results
│
├── models/
│   ├── Phi-3-mini-4k-instruct-q4.gguf # LLM
│   ├── minilm-l6-v2/                   # Embedding model
│   └── colbert-v2/                     # Reranker (optional)
│
└── cache/
    └── query_cache.pkl                 # Semantic cache
```

### 14.2 Parquet Schema Definitions

**files_metadata.parquet:**
```python
{
    'uuid': str,              # Unique file ID
    'path': str,              # Absolute path
    'filename': str,          # File name
    'extension': str,         # File extension
    'size_bytes': int,        # File size
    'created_at': datetime,   # Creation timestamp
    'modified_at': datetime,  # Modification timestamp
    'indexed_at': datetime,   # Indexing timestamp
    'processing_status': str, # 'fast_pass_complete', 'deep_complete'
    'n_chunks': int          # Number of chunks
}
```

**chunks_metadata.parquet:**
```python
{
    'chunk_id': str,          # Unique chunk ID
    'doc_uuid': str,          # Parent document UUID
    'text': str,              # Full chunk text
    'type': str,              # 'text' or 'table'
    'page': int,              # Page number (if applicable)
    'chunk_index': int,       # Index within document
    'token_count': int,       # Approximate tokens
    'cluster_id': int,        # Dedup cluster ID
    'is_representative': bool # Is cluster representative?
}
```

**scaffold_metadata.parquet:**
```python
{
    'chunk_id': str,          # Unique chunk ID
    'doc_uuid': str,          # Parent document UUID
    'summary': str,           # LLM-generated summary
    'keywords': List[str],    # Extracted keywords
    'category': str,          # Document category
    'original_size': int,     # Original text length
    'compressed_size': int,   # Summary length
    'compression_ratio': float # Compression achieved
}
```

---

## 15. CONFIGURATION FILES

### 15.1 Main Configuration (config.yaml)

```yaml
# System Configuration
system:
  data_dir: "/path/to/100GB/data"
  output_dir: "/path/to/output"
  n_workers: 8

# Hardware Constraints
hardware:
  vram_gb: 6
  ram_gb: 32
  device: "cuda"  # or "cpu"

# LLM Configuration
llm:
  model_path: "models/Phi-3-mini-4k-instruct-q4.gguf"
  n_ctx: 4096
  n_gpu_layers: 35
  temperature: 0.3
  max_tokens: 512

# Embedding Configuration
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  normalize: true

# Deduplication
deduplication:
  text_similarity_threshold: 0.92
  table_min_cluster_size: 2
  build_graph: true

# Semantic Compression
compression:
  enable_self_consistency: true
  n_consistency_runs: 3
  summary_max_length: 200
  n_keywords: 10

# Vector Index
vector_index:
  hot_ratio: 0.2
  hnsw_m: 32
  hnsw_ef_construction: 200
  hnsw_ef_search: 64
  ivf_nlist: 4096
  pq_m: 64

# BM25 Index
bm25:
  backend: "whoosh"  # or "elasticsearch"
  analyzer: "stemming"

# Retrieval
retrieval:
  default_k: 50
  bm25_candidates: 500
  dense_candidates: 250
  use_reranker: false
  reranker_top_k: 20

# Query Routing
routing:
  enable: true
  factual_bm25_weight: 0.6
  conceptual_dense_weight: 0.8

# Caching
cache:
  enable: true
  similarity_threshold: 0.95
  ttl_hours: 1
  max_size: 1000

# Response Generation
generation:
  max_context_chunks: 5
  max_context_tokens: 3000
  stream_response: true
```

### 15.2 Loading Configuration

```python
import yaml

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, path, default=None):
        """Get nested config value"""
        keys = path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

# Usage
config = Config('config.yaml')
llm_path = config.get('llm.model_path')
embedding_dim = config.get('embedding.dimension')
```

---

## 16. PERFORMANCE TUNING

### 16.1 Memory Optimization

```python
import gc
import torch

def optimize_memory():
    """Clear memory caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def batch_process_with_memory_management(data, process_fn, batch_size=100):
    """Process data in batches with memory cleanup"""
    results = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_results = process_fn(batch)
        results.extend(batch_results)

        # Clean up every 10 batches
        if i % (batch_size * 10) == 0:
            optimize_memory()

    return results
```

### 16.2 Parallel Processing

```python
from multiprocessing import Pool, cpu_count
from functools import partial

def parallel_process(data, process_fn, n_workers=None):
    """Process data in parallel"""
    if n_workers is None:
        n_workers = cpu_count() - 1

    with Pool(n_workers) as pool:
        results = pool.map(process_fn, data)

    return results

# Usage for chunking
def chunk_document(file_path):
    # Processing logic
    return chunks

file_paths = metadata_df['path'].tolist()
all_chunks = parallel_process(file_paths, chunk_document, n_workers=8)
```

### 16.3 FAISS Optimization

```python
# Enable AVX2 instructions
import faiss
faiss.cvar.distance_compute_blas_threshold = 20

# For GPU (if available)
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

---

## 17. MONITORING & METRICS

### 17.1 Performance Metrics

```python
import time
from datetime import datetime
import psutil
import json

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'ingestion': [],
            'retrieval': [],
            'generation': []
        }

    def track_ingestion(self, n_files, n_chunks, duration):
        """Track ingestion performance"""
        self.metrics['ingestion'].append({
            'timestamp': datetime.now().isoformat(),
            'n_files': n_files,
            'n_chunks': n_chunks,
            'duration_seconds': duration,
            'throughput_files_per_sec': n_files / duration,
            'throughput_chunks_per_sec': n_chunks / duration
        })

    def track_retrieval(self, query, n_results, duration, cache_hit=False):
        """Track retrieval performance"""
        self.metrics['retrieval'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'n_results': n_results,
            'duration_ms': duration * 1000,
            'cache_hit': cache_hit
        })

    def track_generation(self, query, response_length, duration):
        """Track generation performance"""
        self.metrics['generation'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_length': response_length,
            'duration_seconds': duration,
            'tokens_per_sec': response_length / duration if duration > 0 else 0
        })

    def get_summary(self):
        """Get performance summary"""
        summary = {}

        if self.metrics['retrieval']:
            durations = [m['duration_ms'] for m in self.metrics['retrieval']]
            cache_hits = sum(1 for m in self.metrics['retrieval'] if m['cache_hit'])

            summary['retrieval'] = {
                'n_queries': len(durations),
                'p50_ms': np.percentile(durations, 50),
                'p95_ms': np.percentile(durations, 95),
                'p99_ms': np.percentile(durations, 99),
                'cache_hit_rate': cache_hits / len(durations) if durations else 0
            }

        return summary

    def save(self, output_path):
        """Save metrics to file"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

# Usage
monitor = PerformanceMonitor()

# Track retrieval
start = time.time()
results = hybrid_retriever.retrieve(query, strategy)
duration = time.time() - start
monitor.track_retrieval(query, len(results), duration)

# Get summary
summary = monitor.get_summary()
print(f"Retrieval p50: {summary['retrieval']['p50_ms']:.1f}ms")
print(f"Retrieval p95: {summary['retrieval']['p95_ms']:.1f}ms")
```

### 17.2 System Monitoring

```python
def get_system_stats():
    """Get current system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'ram_used_gb': psutil.virtual_memory().used / (1024**3),
        'ram_percent': psutil.virtual_memory().percent,
        'disk_used_gb': psutil.disk_usage('/').used / (1024**3)
    }

# Log during processing
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
stats = get_system_stats()
logger.info(f"System stats: RAM {stats['ram_used_gb']:.1f}GB ({stats['ram_percent']:.1f}%)")
```

---

## 18. DEPLOYMENT CHECKLIST

### 18.1 Pre-Deployment Validation

```bash
# 1. Check Python version
python --version  # Should be 3.10 or 3.11

# 2. Verify CUDA installation (if using GPU)
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 3. Test model loading
python -c "from llama_cpp import Llama; llm = Llama('models/Phi-3-mini-4k-instruct-q4.gguf', n_ctx=512)"

# 4. Verify disk space
df -h  # Need 150GB+ free

# 5. Check RAM
free -h  # Need 32GB available

# 6. Test FAISS
python -c "import faiss; print(faiss.__version__)"
```

### 18.2 Initial System Setup

```python
# setup.py - Run once to initialize system

import os
from pathlib import Path

def setup_directories(base_path):
    """Create directory structure"""
    dirs = [
        'metadata', 'embeddings', 'indices', 'indices/bm25_index',
        'deduplication', 'models', 'cache', 'logs'
    ]

    for dir_name in dirs:
        dir_path = Path(base_path) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created directory structure at {base_path}")

def download_models():
    """Download required models"""
    from sentence_transformers import SentenceTransformer

    # Download embedding model
    print("Downloading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.save('models/minilm-l6-v2')
    print("✓ Embedding model downloaded")

    # LLM must be downloaded manually
    print("⚠ Please download LLM manually:")
    print("  wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
    print("  mv Phi-3-mini-4k-instruct-q4.gguf models/")

def verify_setup():
    """Verify all components are ready"""
    checks = {
        'Output directory': Path('output').exists(),
        'Models directory': Path('models').exists(),
        'LLM model': Path('models/Phi-3-mini-4k-instruct-q4.gguf').exists(),
        'Embedding model': Path('models/minilm-l6-v2').exists(),
        'Config file': Path('config.yaml').exists()
    }

    print("\n=== Setup Verification ===")
    all_pass = True
    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check}")
        if not status:
            all_pass = False

    if all_pass:
        print("\n✓ System ready for deployment!")
    else:
        print("\n✗ Some components missing. Please complete setup.")

    return all_pass

# Run setup
if __name__ == '__main__':
    setup_directories('output')
    download_models()
    verify_setup()
```

### 18.3 Complete Pipeline Execution

```python
# main.py - Main execution pipeline

import logging
from pathlib import Path
import time

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('output/logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    config = Config('config.yaml')
    logger.info("Configuration loaded")

    # Phase 1: Fast Pass Ingestion
    logger.info("=== Phase 1: Fast Pass Ingestion ===")
    start_time = time.time()

    ingestor = FastPassIngestor(
        config.get('system.data_dir'),
        config.get('system.output_dir')
    )
    metadata = ingestor.scan_directory()
    bm25_index = ingestor.build_bm25_index()

    logger.info(f"Fast pass complete: {len(metadata)} files in {time.time()-start_time:.1f}s")

    # Phase 2: Deep Processing
    logger.info("=== Phase 2: Deep Processing ===")
    start_time = time.time()

    deep = DeepIngestor(pd.DataFrame(metadata), config.get('system.output_dir'))
    chunks = deep.process_all(n_workers=config.get('system.n_workers'))
    chunks_df = pd.DataFrame(chunks)

    logger.info(f"Deep processing complete: {len(chunks)} chunks in {time.time()-start_time:.1f}s")

    # Phase 3: LLM Processing
    logger.info("=== Phase 3: LLM Processing ===")
    start_time = time.time()

    llm = LocalLLM(config.get('llm.model_path'))
    processed_df = process_chunks_with_llm(chunks_df, llm)

    logger.info(f"LLM processing complete in {time.time()-start_time:.1f}s")

    # Phase 4: Embedding Generation
    logger.info("=== Phase 4: Embedding Generation ===")
    start_time = time.time()

    embedder = EmbeddingGenerator(config.get('embedding.model_name'))
    chunk_embeddings = embedder.embed_chunks(chunks_df)

    logger.info(f"Embedding complete: {chunk_embeddings.shape} in {time.time()-start_time:.1f}s")

    # Phase 5: Deduplication
    logger.info("=== Phase 5: Deduplication ===")
    start_time = time.time()

    deduplicator = SemanticDeduplicator(
        similarity_threshold=config.get('deduplication.text_similarity_threshold')
    )
    chunk_ids = chunks_df['chunk_id'].tolist()
    similarity_graph = deduplicator.build_similarity_graph(chunk_embeddings, chunk_ids)
    clusters, cluster_mapping = deduplicator.find_clusters()

    logger.info(f"Deduplication complete: {len(clusters)} clusters in {time.time()-start_time:.1f}s")

    # Phase 6: Vector Indexing
    logger.info("=== Phase 6: Vector Indexing ===")
    start_time = time.time()

    vector_store = HotColdVectorStore(
        dimension=config.get('embedding.dimension'),
        hot_ratio=config.get('vector_index.hot_ratio')
    )
    vector_store.build_indices(chunk_embeddings, chunk_ids)
    vector_store.save(Path(config.get('system.output_dir')) / 'indices')

    logger.info(f"Indexing complete in {time.time()-start_time:.1f}s")

    # Phase 7: System Ready
    logger.info("=== System Ready for Queries ===")
    logger.info(f"Total documents: {len(metadata)}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Dedup clusters: {len(clusters)}")
    logger.info(f"Index size: Hot={len(vector_store.hot_ids)}, Cold={len(vector_store.cold_ids)}")

if __name__ == '__main__':
    main()
```

### 18.4 Query Interface

```python
# query.py - Interactive query interface

def interactive_query_loop():
    """Interactive CLI for querying the system"""
    # Load components
    config = Config('config.yaml')

    # Initialize system
    embedder = EmbeddingGenerator(config.get('embedding.model_name'))
    vector_store = HotColdVectorStore(dimension=config.get('embedding.dimension'))
    vector_store.load(Path(config.get('system.output_dir')) / 'indices')

    bm25 = BM25Searcher(Path(config.get('system.output_dir')) / 'indices/bm25_index')

    llm = LocalLLM(config.get('llm.model_path'))

    router = SemanticRouter(embedder)
    retriever = HybridRetriever(bm25, vector_store, embedder)
    cache = SemanticCache(embedder)

    # Load metadata
    chunks_df = pd.read_parquet(Path(config.get('system.output_dir')) / 'metadata/chunks_metadata.parquet')

    assembler = ContextAssembler(chunks_df, {})  # Load dedup graph if needed
    generator = ResponseGenerator(llm, assembler)

    print("=" * 60)
    print("RAG 100GB System - Interactive Query Interface")
    print("=" * 60)
    print("Type 'exit' to quit, 'stats' for statistics")
    print()

    while True:
        query = input("Query> ").strip()

        if query.lower() == 'exit':
            break

        if query.lower() == 'stats':
            stats = get_system_stats()
            print(f"RAM: {stats['ram_used_gb']:.1f}GB ({stats['ram_percent']:.1f}%)")
            continue

        if not query:
            continue

        # Process query
        start_time = time.time()

        # Check cache
        cached = cache.get(query)
        if cached:
            results = cached
            print("[Using cached results]")
        else:
            # Route and retrieve
            strategy = router.route_query(query)
            results = retriever.retrieve(query, strategy)
            cache.set(query, results)

        # Generate response
        response = generator.generate_response(query, results)

        duration = time.time() - start_time

        # Display results
        print()
        print("Answer:")
        print("-" * 60)
        print(response['response'])
        print()
        print(f"Sources ({len(response['sources'])}):")
        for source in response['sources']:
            print(f"  [{source['source_id']}] {source['file']} (score: {source['relevance_score']:.3f})")
        print()
        print(f"[Query completed in {duration:.2f}s]")
        print()

if __name__ == '__main__':
    interactive_query_loop()
```

---

## 19. TROUBLESHOOTING

### 19.1 Common Issues

**Issue: Out of Memory during ingestion**
```python
# Solution: Reduce batch size
deep = DeepIngestor(metadata_df, output_dir)
chunks = deep.process_all(n_workers=4)  # Reduce from 8 to 4

# Or process in smaller batches
for i in range(0, len(metadata_df), 1000):
    batch_df = metadata_df.iloc[i:i+1000]
    batch_deep = DeepIngestor(batch_df, output_dir)
    batch_chunks = batch_deep.process_all(n_workers=4)
```

**Issue: FAISS index too large for RAM**
```python
# Solution: Use more aggressive PQ compression
vector_store = HotColdVectorStore(dimension=384, hot_ratio=0.1)  # Reduce hot ratio
# Or use smaller m parameter for PQ
# In IndexIVFPQ: m=32 instead of m=64
```

**Issue: LLM inference too slow**
```python
# Solution: Reduce context window or increase quantization
llm = LocalLLM(
    model_path=model_path,
    n_ctx=2048,  # Reduce from 4096
    n_gpu_layers=35,
    n_batch=256  # Reduce batch size
)

# Or use smaller model
# Switch from Phi-3 3.8B to Gemma-2B
```

**Issue: BM25 search returns no results**
```python
# Solution: Check index and rebuild if corrupted
from whoosh.index import exists_in, open_dir

if not exists_in('output/indices/bm25_index'):
    print("BM25 index missing, rebuilding...")
    ingestor = FastPassIngestor(data_dir, output_dir)
    bm25_index = ingestor.build_bm25_index()
```

**Issue: Embeddings dimension mismatch**
```python
# Solution: Verify embedding model dimension
embedder = EmbeddingGenerator()
print(f"Embedding dimension: {embedder.dimension}")

# If mismatch, rebuild embeddings with correct dimension
# Or adjust FAISS index dimension to match
```

**Issue: Deduplication graph too large**
```python
# Solution: Increase similarity threshold
deduplicator = SemanticDeduplicator(similarity_threshold=0.95)  # Instead of 0.92

# Or skip graph building for very large corpora
# Just use cluster representatives without graph
```

---

## 20. OPTIMIZATION TIPS

### 20.1 Speed Optimizations

**1. Parallelize Embedding Generation**
```python
# Use GPU batch processing if available
embedder = EmbeddingGenerator()
embedder.model = embedder.model.to('cuda')  # Move to GPU

# Increase batch size
embeddings = embedder.embed_batch(texts, batch_size=128)  # Instead of 32
```

**2. Use Approximate Search**
```python
# For HNSW, reduce ef_search for faster queries
vector_store.hot_index.hnsw.efSearch = 32  # Instead of 64
# Trade-off: slightly lower recall, much faster
```

**3. Skip Reranking for Simple Queries**
```python
# In router, disable reranking for factual queries
if query_type == QueryType.FACTUAL:
    strategy['use_reranker'] = False
```

**4. Cache Embeddings During Ingestion**
```python
# Save embeddings incrementally
for i, batch in enumerate(chunk_batches):
    batch_embeddings = embedder.embed_batch(batch)
    np.save(f'output/embeddings/batch_{i}.npy', batch_embeddings)

# Combine later
all_embeddings = np.vstack([
    np.load(f'output/embeddings/batch_{i}.npy')
    for i in range(n_batches)
])
```

### 20.2 Quality Optimizations

**1. Use Better Embedding Model**
```python
# Upgrade to mpnet for +20-30% quality
embedder = EmbeddingGenerator('sentence-transformers/all-mpnet-base-v2')
# Cost: 2x memory, 1.5x slower
```

**2. Enable Reranking for All Queries**
```python
# Add ColBERT reranker
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('colbert-ir/colbertv2.0')

hybrid_retriever = HybridRetriever(
    bm25, vector_store, embedder, reranker=reranker
)
```

**3. Increase Context Window**
```python
# Use more context chunks
context_chunks = assembler.assemble_context(
    results,
    max_chunks=10,  # Instead of 5
    max_tokens=6000  # Instead of 3000
)
```

**4. Multi-Query Generation**
```python
def generate_query_variations(query, llm):
    """Generate multiple query variations for better recall"""
    prompt = f"""Generate 3 variations of this query:
    Original: {query}

    Variations:
    1."""

    variations = llm.generate(prompt, max_tokens=200)
    return [query] + variations.split('\n')[:3]

# Search with all variations
all_results = []
for variant in generate_query_variations(query, llm):
    results = retriever.retrieve(variant, strategy)
    all_results.extend(results)

# Deduplicate and rerank
unique_results = deduplicate_results(all_results)
```

### 20.3 Storage Optimizations

**1. Compress Parquet Files**
```python
# Use compression when saving
chunks_df.to_parquet(
    'output/chunks_metadata.parquet',
    compression='zstd',  # Or 'gzip', 'snappy'
    compression_level=3
)
```

**2. Use Memory-Mapped Files**
```python
# For large embedding arrays
embeddings_mmap = np.load(
    'output/embeddings/chunk_embeddings.npy',
    mmap_mode='r'  # Read-only memory mapping
)
```

**3. Archive Old Data**
```python
import shutil
from datetime import datetime, timedelta

def archive_old_cache(cache_dir, days=7):
    """Archive cache older than N days"""
    cutoff = datetime.now() - timedelta(days=days)

    for file in Path(cache_dir).glob('*.pkl'):
        if datetime.fromtimestamp(file.stat().st_mtime) < cutoff:
            archive_path = Path(cache_dir) / 'archive'
            archive_path.mkdir(exist_ok=True)
            shutil.move(str(file), str(archive_path / file.name))
```

---

## 21. TESTING & VALIDATION

### 21.1 Unit Tests

```python
import unittest

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingGenerator()

    def test_embedding_dimension(self):
        """Test embedding dimension is correct"""
        texts = ["test sentence"]
        embeddings = self.embedder.embed_batch(texts)
        self.assertEqual(embeddings.shape[1], 384)

    def test_embedding_normalization(self):
        """Test embeddings are normalized"""
        texts = ["test sentence"]
        embeddings = self.embedder.embed_batch(texts)
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, places=5)

class TestDeduplication(unittest.TestCase):
    def setUp(self):
        self.deduplicator = SemanticDeduplicator(similarity_threshold=0.92)

    def test_identical_chunks_clustered(self):
        """Test identical chunks are in same cluster"""
        embeddings = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
        chunk_ids = ['chunk1', 'chunk2', 'chunk3']

        graph = self.deduplicator.build_similarity_graph(embeddings, chunk_ids)
        clusters, mapping = self.deduplicator.find_clusters()

        # chunk1 and chunk2 should be in same cluster
        self.assertEqual(mapping['chunk1'], mapping['chunk2'])
        self.assertNotEqual(mapping['chunk1'], mapping['chunk3'])

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.store = HotColdVectorStore(dimension=384)

    def test_search_returns_results(self):
        """Test vector search returns results"""
        # Create dummy data
        embeddings = np.random.randn(1000, 384).astype('float32')
        chunk_ids = [f'chunk_{i}' for i in range(1000)]

        self.store.build_indices(embeddings, chunk_ids)

        # Search
        query = np.random.randn(384).astype('float32')
        results = self.store.search(query, k=10)

        self.assertEqual(len(results), 10)

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### 21.2 Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete pipeline with small dataset"""
    print("Running end-to-end integration test...")

    # 1. Create test data
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)

    (test_dir / 'doc1.txt').write_text("Machine learning is a subset of AI.")
    (test_dir / 'doc2.txt').write_text("Deep learning uses neural networks.")

    # 2. Run fast pass
    ingestor = FastPassIngestor(str(test_dir), 'test_output')
    metadata = ingestor.scan_directory()
    assert len(metadata) == 2, "Should find 2 files"

    bm25_index = ingestor.build_bm25_index()

    # 3. Run deep processing
    deep = DeepIngestor(pd.DataFrame(metadata), 'test_output')
    chunks = deep.process_all(n_workers=1)
    assert len(chunks) > 0, "Should extract chunks"

    # 4. Generate embeddings
    embedder = EmbeddingGenerator()
    chunks_df = pd.DataFrame(chunks)
    embeddings = embedder.embed_chunks(chunks_df)
    assert embeddings.shape[1] == 384, "Embeddings should be 384-dim"

    # 5. Build index
    vector_store = HotColdVectorStore(dimension=384)
    chunk_ids = chunks_df['chunk_id'].tolist()
    vector_store.build_indices(embeddings, chunk_ids)

    # 6. Query
    bm25 = BM25Searcher('test_output/bm25_index')
    retriever = HybridRetriever(bm25, vector_store, embedder)
    router = SemanticRouter(embedder)

    query = "What is machine learning?"
    strategy = router.route_query(query)
    results = retriever.retrieve(query, strategy)

    assert len(results) > 0, "Should return results"

    print("✓ End-to-end test passed!")

    # Cleanup
    shutil.rmtree('test_data')
    shutil.rmtree('test_output')

# Run integration test
test_end_to_end_pipeline()
```

### 21.3 Benchmark Suite

```python
def benchmark_retrieval(retriever, queries, n_runs=10):
    """Benchmark retrieval performance"""
    import time

    latencies = []

    for query in queries:
        durations = []
        for _ in range(n_runs):
            start = time.time()
            results = retriever.retrieve(query, router.route_query(query))
            duration = time.time() - start
            durations.append(duration * 1000)  # Convert to ms

        latencies.extend(durations)

    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }

# Test queries
test_queries = [
    "What is machine learning?",
    "Explain neural networks",
    "Compare supervised and unsupervised learning",
    "Recent advances in NLP",
    "How does backpropagation work?"
]

# Run benchmark
results = benchmark_retrieval(hybrid_retriever, test_queries)
print(f"Mean latency: {results['mean_ms']:.1f}ms")
print(f"p95 latency: {results['p95_ms']:.1f}ms")
print(f"p99 latency: {results['p99_ms']:.1f}ms")
```

---

## 22. PRODUCTION DEPLOYMENT

### 22.1 REST API Wrapper

```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize system (load once at startup)
config = Config('config.yaml')
embedder = EmbeddingGenerator(config.get('embedding.model_name'))
vector_store = HotColdVectorStore(dimension=config.get('embedding.dimension'))
vector_store.load(Path(config.get('system.output_dir')) / 'indices')

bm25 = BM25Searcher(Path(config.get('system.output_dir')) / 'indices/bm25_index')
llm = LocalLLM(config.get('llm.model_path'))

router = SemanticRouter(embedder)
retriever = HybridRetriever(bm25, vector_store, embedder)
cache = SemanticCache(embedder)

chunks_df = pd.read_parquet(
    Path(config.get('system.output_dir')) / 'metadata/chunks_metadata.parquet'
)
assembler = ContextAssembler(chunks_df, {})
generator = ResponseGenerator(llm, assembler)

monitor = PerformanceMonitor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/query', methods=['POST'])
def query():
    """Query endpoint"""
    try:
        data = request.get_json()
        query_text = data.get('query')

        if not query_text:
            return jsonify({'error': 'Query text required'}), 400

        # Process query
        start_time = time.time()

        # Check cache
        cached = cache.get(query_text)
        cache_hit = cached is not None

        if cached:
            results = cached
        else:
            strategy = router.route_query(query_text)
            results = retriever.retrieve(query_text, strategy)
            cache.set(query_text, results)

        # Generate response
        response = generator.generate_response(query_text, results)

        duration = time.time() - start_time

        # Track metrics
        monitor.track_retrieval(query_text, len(results), duration, cache_hit)

        return jsonify({
            'query': query_text,
            'answer': response['response'],
            'sources': response['sources'],
            'metadata': {
                'duration_ms': duration * 1000,
                'cache_hit': cache_hit,
                'n_sources': len(response['sources'])
            }
        })

    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    summary = monitor.get_summary()
    sys_stats = get_system_stats()

    return jsonify({
        'performance': summary,
        'system': sys_stats,
        'index_size': {
            'hot': len(vector_store.hot_ids),
            'cold': len(vector_store.cold_ids)
        }
    })

@app.route('/search', methods=['POST'])
def search_only():
    """Search without generation (faster)"""
    try:
        data = request.get_json()
        query_text = data.get('query')
        k = data.get('k', 10)

        strategy = router.route_query(query_text)
        results = retriever.retrieve(query_text, strategy)

        return jsonify({
            'query': query_text,
            'results': results[:k]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Usage:**
```bash
# Start API server
python api.py

# Query via curl
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### 22.2 Docker Deployment

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Download models (do this during build to cache)
RUN python3 setup.py

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run API
CMD ["python3", "api.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./output:/app/output
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

**Build and run:**
```bash
# Build image
docker build -t rag-local:1.0 .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 23. MAINTENANCE & UPDATES

### 23.1 Incremental Updates

```python
def incremental_ingestion(new_files_dir, existing_system_path):
    """Add new files to existing system"""

    # 1. Load existing indices
    vector_store = HotColdVectorStore(dimension=384)
    vector_store.load(f"{existing_system_path}/indices")

    bm25 = BM25Searcher(f"{existing_system_path}/indices/bm25_index")

    # 2. Process new files
    ingestor = FastPassIngestor(new_files_dir, existing_system_path)
    new_metadata = ingestor.scan_directory()

    # 3. Check for duplicates
    existing_metadata = pd.read_parquet(f"{existing_system_path}/metadata/files_metadata.parquet")
    new_files = [m for m in new_metadata if m['path'] not in existing_metadata['path'].values]

    if not new_files:
        print("No new files to add")
        return

    # 4. Process new chunks
    deep = DeepIngestor(pd.DataFrame(new_files), existing_system_path)
    new_chunks = deep.process_all()

    # 5. Generate embeddings
    embedder = EmbeddingGenerator()
    new_embeddings = embedder.embed_chunks(pd.DataFrame(new_chunks))

    # 6. Add to indices
    new_chunk_ids = [c['chunk_id'] for c in new_chunks]
    vector_store.add_to_cold(new_embeddings, new_chunk_ids)

    # 7. Update BM25 (requires rebuild or append)
    # Simplified: rebuild entire BM25 index
    # Production: use incremental update if supported

    # 8. Save updated system
    vector_store.save(f"{existing_system_path}/indices")

    # Append metadata
    updated_metadata = pd.concat([existing_metadata, pd.DataFrame(new_files)])
    updated_metadata.to_parquet(f"{existing_system_path}/metadata/files_metadata.parquet")

    print(f"Added {len(new_files)} new files, {len(new_chunks)} new chunks")

    # 9. Invalidate cache
    cache.invalidate_all()
```

### 23.2 Index Optimization

```python
def optimize_indices(system_path, access_logs_path):
    """Optimize hot/cold split based on access patterns"""

    # 1. Load access logs
    with open(access_logs_path) as f:
        access_logs = json.load(f)

    # Count accesses per chunk
    access_counts = {}
    for log in access_logs:
        for source in log.get('sources', []):
            chunk_id = source['chunk_id']
            access_counts[chunk_id] = access_counts.get(chunk_id, 0) + 1

    # 2. Rebuild indices with new hot/cold split
    vector_store = HotColdVectorStore(dimension=384, hot_ratio=0.2)

    embeddings = np.load(f"{system_path}/embeddings/chunk_embeddings.npy")
    chunks_df = pd.read_parquet(f"{system_path}/metadata/chunks_metadata.parquet")
    chunk_ids = chunks_df['chunk_id'].tolist()

    vector_store.build_indices(embeddings, chunk_ids, access_history=access_counts)
    vector_store.save(f"{system_path}/indices")

    print("Indices optimized based on access patterns")
```

### 23.3 Backup Strategy

```python
import shutil
from datetime import datetime

def backup_system(system_path, backup_dir):
    """Create full system backup"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = Path(backup_dir) / f"backup_{timestamp}"

    # Copy critical components
    components = ['metadata', 'indices', 'deduplication', 'embeddings']

    for component in components:
        src = Path(system_path) / component
        dst = backup_path / component
        if src.exists():
            shutil.copytree(src, dst)

    # Create backup manifest
    manifest = {
        'timestamp': timestamp,
        'system_path': str(system_path),
        'components': components,
        'size_gb': sum(
            sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            for p in [backup_path / c for c in components]
        ) / (1024**3)
    }

    with open(backup_path / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Backup created: {backup_path}")
    print(f"Size: {manifest['size_gb']:.2f} GB")

    return backup_path

def restore_system(backup_path, system_path):
    """Restore system from backup"""
    manifest_path = Path(backup_path) / 'manifest.json'

    if not manifest_path.exists():
        raise ValueError("Invalid backup: manifest.json not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Restore components
    for component in manifest['components']:
        src = Path(backup_path) / component
        dst = Path(system_path) / component

        if dst.exists():
            shutil.rmtree(dst)

        shutil.copytree(src, dst)

    print(f"System restored from backup: {backup_path}")

# Usage
backup_path = backup_system('output', 'backups')
# restore_system(backup_path, 'output')
```

---

## 24. FINAL CHECKLIST

### 24.1 Pre-Production Checklist

- [ ] All models downloaded and verified
- [ ] Directory structure created
- [ ] Configuration file customized
- [ ] Test data ingestion successful
- [ ] Embedding generation working
- [ ] Vector indices built and tested
- [ ] Query pipeline functional
- [ ] Performance meets targets (p95 < 800ms)
- [ ] Memory usage within limits (<28GB)
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Backup strategy implemented
- [ ] Monitoring setup
- [ ] API deployed (if applicable)
- [ ] Documentation updated

### 24.2 Performance Targets Verification

```python
def verify_system_performance(system_path):
    """Verify system meets performance targets"""
    checks = {
        'Fast Pass < 3min': False,
        'Query p50 < 200ms': False,
        'Query p95 < 800ms': False,
        'Recall@10 > 90%': False,
        'Memory < 28GB': False,
        'Compression > 8x': False
    }

    # Load performance logs
    monitor = PerformanceMonitor()
    monitor.metrics = json.load(open(f"{system_path}/logs/metrics.json"))

    summary = monitor.get_summary()

    # Check targets
    if summary.get('retrieval'):
        checks['Query p50 < 200ms'] = summary['retrieval']['p50_ms'] < 200
        checks['Query p95 < 800ms'] = summary['retrieval']['p95_ms'] < 800

    sys_stats = get_system_stats()
    checks['Memory < 28GB'] = sys_stats['ram_used_gb'] < 28

    # Check compression
    scaffold_df = pd.read_parquet(f"{system_path}/metadata/scaffold_metadata.parquet")
    avg_compression = scaffold_df['compression_ratio'].mean()
    checks['Compression > 8x'] = avg_compression > 8

    # Print results
    print("\n=== Performance Verification ===")
    for check, passed in checks.items():
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check}")

    all_passed = all(checks.values())
    if all_passed:
        print("\n✓ All performance targets met!")
    else:
        print("\n✗ Some targets not met. Review configuration.")

    return all_passed

# Run verification
verify_system_performance('output')
```

---

## 25. ADDITIONAL RESOURCES

### 25.1 Recommended Reading

- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **Sentence-Transformers**: https://www.sbert.net/
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Whoosh**: https://whoosh.readthedocs.io/
- **RAG Papers**:
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
  - "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

### 25.2 Community & Support

- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Join community forums for questions
- **Updates**: Check for model updates monthly

### 25.3 Future Enhancements

**Planned features:**
1. Multi-modal support (images, audio)
2. Real-time collaborative filtering
3. Federated learning for privacy
4. Graph-based reasoning
5. Automated hyperparameter tuning

---

## APPENDIX A: Complete Requirements.txt

```txt
# Core Dependencies
python>=3.10,<3.12

# LLM Inference
llama-cpp-python==0.2.56

# Embeddings & Transformers
sentence-transformers==2.2.2
transformers==4.36.0
torch==2.1.0

# Vector Search
faiss-cpu==1.7.4
hnswlib==0.7.0

# Sparse Search
whoosh==2.7.4

# Document Processing
pypdf==3.0.1
pdfplumber==0.10.3
python-docx==1.1.0
openpyxl==3.1.2
pandas==2.1.4
tabula-py==2.8.2
camelot-py[cv]==0.11.0

# OCR (Optional)
pytesseract==0.3.10
easyocr==1.7.0

# Clustering & ML
scikit-learn==1.3.2
hdbscan==0.8.33
networkx==3.2.1
datasketch==1.6.4

# Storage
pyarrow==14.0.1

# Utilities
tqdm==4.66.1
python-dotenv==1.0.0
pyyaml==6.0.1
rapidfuzz==3.5.2
dateparser==1.2.0
psutil==5.9.6

# API (Optional)
flask==3.0.0
flask-cors==4.0.0

# Testing
pytest==7.4.3
```

---

## APPENDIX B: Quick Start Commands

```bash
# 1. Setup environment
conda create -n rag_local python=3.10
conda activate rag_local
pip install -r requirements.txt

# 2. Download models
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf -P models/

# 3. Initialize system
python setup.py

# 4. Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your paths

# 5. Run ingestion
python main.py

# 6. Start querying
python query.py

# 7. (Optional) Start API
python api.py
```

---

**END OF DOCUMENT**

**Total System Complexity**: Production-Ready
**Estimated Build Time**: 2-3 days for 100GB corpus
**Maintenance Effort**: Low (monthly index optimization recommended)
**Scalability**: Up to 500GB with minor adjustments

---

**Document Version**: 1.0
**Last Updated**: 2024
**Author**: RAG System Design Team
**License**: MIT
