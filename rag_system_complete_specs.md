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
pip install PyPDF2==3.0.1
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
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
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
sparse_