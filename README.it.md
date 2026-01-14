[🇬🇧 English](README.md) | [🇮🇹 Italiano](README.it.md) | [🇨🇳 中文](README.zh.md)

<div align="center">

# 🧊 CUBO
### Il RAG Locale di Livello Industriale

**Cerca nei documenti aziendali direttamente dal tuo laptop. 100% Offline. Conforme al GDPR.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: Pytest](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

</div>

---

**CUBO** è un sistema di Retrieval-Augmented Generation (RAG) costruito per ambienti **privacy-first**. Funziona interamente sulla tua macchina locale, ingerendo gigabyte di documenti senza inviare un singolo byte al cloud.

A differenza dei semplici wrapper RAG, CUBO è ingegnerizzato per **hardware limitato** (laptop da 16GB RAM) e per le **lingue europee**.

## ✨ Perché CUBO?

| Funzionalità | Perché è importante |
| :--- | :--- |
| **🚀 Modalità Laptop** | La gestione intelligente delle risorse (Float16, Caricamento Pigro) permette di gestire indici massivi su **8GB/16GB RAM**. |
| **🌍 Cuore Europeo** | Tokenizzazione avanzata per **Italiano, Francese, Tedesco e Spagnolo**. Abbina automaticamente "gatto" con "gatti". |
| **🛡️ 100% Offline** | Niente OpenAI. Niente Pinecone. Niente Weaviate. I tuoi dati **non lasciano mai** il tuo SSD. |
| **⚡ Streaming** | La generazione dei token in tempo reale sembra istantanea, anche su hardware solo CPU. |
| **🧠 Ingestione Smart** | L'ingestione streaming in Parquet gestisce **corpora da 50GB+** senza saturare la RAM. |

## 🚀 Avvio Rapido

**Windows (PowerShell):**
```powershell
.\run_local.ps1
```
*Questo script controlla Python, Node.js e Ollama, scarica i modelli necessari (~2GB) e avvia l'applicazione.*

**Installazione Manuale:**
```bash
pip install -r requirements.txt
python scripts/start_fullstack.py --mode laptop
```

## 📚 Documentazione

Guide dettagliate per sviluppatori e ricercatori:

- **[Guida all'Installazione](docs/API_INTEGRATION.md)** - Istruzioni complete di setup.
- **[Architettura & Ottimizzazione](docs/optimization/resource_architecture.md)** - Come abbiamo risparmiato il 50% di RAM.
- **[Benchmark](docs/eval/evaluation_antigravity.md)** - Recall@10, nDCG e statistiche di velocità.
- **[Paper Scientifico](paper/paper.pdf)** - La teoria accademica dietro CUBO.

## 🛠️ Architettura

CUBO utilizza una strategia di **Retrieval Ibrido a Livelli**:
1.  **Ingestione:** I documenti vengono suddivisi (Structure-Aware) e trasmessi su disco in streaming.
2.  **Indicizzazione:** I vettori vengono quantizzati (Float16) e salvati su SQLite (Metadata) + FAISS (Ricerca).
3.  **Retrieval:** **Reciprocal Rank Fusion (RRF)** combina i punteggi BM25 (Parole chiave) ed Embedding (Semantico).
4.  **Generazione:** LLM locale (Llama 3, Mistral) via Ollama genera la risposta con citazioni.

## 🧪 Valutazione

Crediamo nel misurare, non nell'indovinare.
*   **Recall@10:** 0.96 (Politica), 0.82 (Dominio Misto).
*   **Latenza:** < 300ms per query (cache).
*   **Ingestione:** ~150 pagine/secondo.

Vedi [examples/04_evaluation_benchmarking.ipynb](examples/04_evaluation_benchmarking.ipynb) per eseguire i tuoi benchmark.

## 🤝 Contribuire

I contributi sono benvenuti! Vedi [CONTRIBUTING.md](docs/CONTRIBUTING.md) per dettagli sul nostro codice di condotta e processo di sviluppo.

---

<div align="center">
  <sub>Costruito con ❤️ per Privacy ed Efficienza.</sub>
</div>
