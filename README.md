[🇬🇧 English](README.md) | [🇮🇹 Italiano](README.it.md) | [🇨🇳 中文](README.zh.md)

<div align="center">

# 🧊 CUBO
### The Industrial-Grade Local RAG

**Run enterprise document search on a consumer laptop. 100% Offline. GDPR Compliant.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: Pytest](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03731-b31b1b.svg)](https://arxiv.org/abs/2602.03731)
[![Stars](https://img.shields.io/github/stars/PaoloAstrino/CUBO?style=social)](https://github.com/PaoloAstrino/CUBO)

</div>

<!-- Demo / GIF placeholder: replace XXXXX with the real YouTube id or add assets/demo.gif
[![Watch the 90s demo](https://img.youtube.com/vi/XXXXX/0.jpg)](https://www.youtube.com/watch?v=XXXXX)
_90s demo: drag 5 GB contracts → ingest → query "clausola recesso" → answer with citation_  
_or_ `![demo](assets/demo.svg)`
-->

---

**CUBO** is a Retrieval-Augmented Generation (RAG) system built for **privacy-first** environments. It runs entirely on your local machine, ingesting gigabytes of documents without sending a single byte to the cloud.

Unlike simple RAG wrappers, CUBO is engineered for **constrained hardware** (16GB RAM laptops) and **European languages**.

## ✨ Why CUBO?

| Feature | Why it matters |
| :--- | :--- |
| **🚀 Laptop Mode** | Intelligent resource management (Float16, Lazy Loading) lets you run massive indexes on **8GB/16GB RAM**. |
| **🌍 European Core** | Advanced tokenization for **Italian, French, German, and Spanish**. Matches "gatto" with "gatti" automatically. |
| **🛡️ 100% Offline** | No OpenAI. No Pinecone. No Weaviate. Your data **never** leaves your SSD. |
| **⚡ Streaming** | Real-time token generation feels instant, even on CPU-only hardware. |
| **🧠 Smart Ingestion** | Streaming Parquet ingestion handles **50GB+ corpora** without crashing RAM. |

## 🚀 Quick Start

**Windows (PowerShell):**
```powershell
.\run_local.ps1
```
*This script will check for Python, Node.js, and Ollama, download the required models (~2GB), and launch the app.*

**Manual Install:**
```bash
pip install -r requirements.txt
python scripts/start_fullstack.py --mode laptop
```

## Download & Run

[![Latest Release](https://img.shields.io/github/v/release/PaoloAstrino/CUBO?color=green)](https://github.com/PaoloAstrino/CUBO/releases/latest)

- Windows: [CUBO.exe](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO.exe) (~180 MB)  
- Linux: [CUBO_linux](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO_linux) (PyInstaller)

## 📚 Documentation

Detailed guides for developers and researchers:

- **[Installation Guide](docs/API_INTEGRATION.md)** - Full setup instructions.
- **[Architecture & Optimization](docs/optimization/resource_architecture.md)** - How we saved 50% RAM.
- **[Benchmarks](docs/eval/evaluation_antigravity.md)** - Recall@10, nDCG, and speed stats.
- **[Scientific Paper](paper/paper.pdf)** - The academic theory behind CUBO.

## 🛠️ Architecture

CUBO uses a **Tiered Hybrid Retrieval** strategy:
1.  **Ingestion:** Documents are chunked (Structure-Aware) and streamed to disk.
2.  **Indexing:** Vectors are quantized (Float16) and stored in SQLite (Metadata) + FAISS (Search).
3.  **Retrieval:** **Reciprocal Rank Fusion (RRF)** combines BM25 (Keywords) and Embedding (Semantic) scores.
4.  **Generation:** Local LLM (Llama 3, Mistral) via Ollama generates the answer with citations.

## 💾 Memory Efficiency

CUBO is engineered for **O(1) memory scaling** during document ingestion. Unlike naive approaches that accumulate chunks in RAM, CUBO uses:
-   **Streaming Shards:** Documents are processed in small batches and flushed to Parquet shards.
-   **Deterministic Cleanup:** Explicit garbage collection triggers after each batch to prevent heap fragmentation.
-   **Empirical Validation:** Tested on 0.05GB to 1GB corpora (20× increase) with a constant **30–44 MB delta** in RSS usage.

This ensures you can ingest 50GB+ corpora on a standard 16GB laptop without system lag or crashes.


## 🧪 Evaluation

We believe in measuring, not guessing.
*   **Recall@10:** 0.96 (Politics), 0.82 (Cross-Domain).
*   **Latency:** < 300ms per query (cached).
*   **Ingestion:** ~150 pages/second.

## Real Benchmarks (embedding-gemma-300m, 16 GB laptop)

| Dataset           | Domain     | Recall@10 | Verdict        |
|-------------------|------------|-----------|----------------|
| UltraDomain-Legal | Legal      | 0.48      | ⭐ Strong       |
| Politics          | Structured | 0.97      | 🚀 Perfect     |
| NFCorpus          | Medical    | 0.17      | ⚠️ Domain bias |
| RAGBench-full     | Mixed hard | 0.30      | ⭐ Industry ok  |

_Caption: Strong on structured legal text (our main use-case), weaker on specialized jargon (addressable with router)._ 

See [examples/04_evaluation_benchmarking.ipynb](examples/04_evaluation_benchmarking.ipynb) to run your own benchmarks.

## Who is CUBO for?

- Italian law firms that cannot upload case files to the cloud (89% according to our survey)
- Medical practitioners with sensitive patient records
- Independent researchers who want local RAG without AWS bills
- Anyone with just a 16 GB laptop who wants absolute privacy

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and development process.

---

<div align="center">
  <sub>Built with ❤️ for Privacy and Efficiency.</sub>
</div>
