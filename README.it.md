[🇬🇧 English](README.md) | [🇮🇹 Italiano](README.it.md) | [🇨🇳 中文](README.zh.md)

<div align="center">

# 🧊 CUBO
### Il RAG Locale di Livello Industriale

**Cerca nei documenti aziendali direttamente dal tuo laptop. 100% Offline. Conforme al GDPR.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: Pytest](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![Stars](https://img.shields.io/github/stars/PaoloAstrino/CUBO?style=social)](https://github.com/PaoloAstrino/CUBO)

</div>

<!-- Demo / GIF placeholder: sostituisci XXXXX con l'ID YouTube reale o aggiungi assets/demo.gif
[![Guarda la demo di 90s](https://img.youtube.com/vi/XXXXX/0.jpg)](https://www.youtube.com/watch?v=XXXXX)
_Demo 90s: trascina 5 GB di contratti → ingestione → query "clausola recesso" → risposta con citazione_  
_oppure_ `![demo](assets/demo.svg)`
-->

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

## Scarica ed Esegui

[![Ultima Release](https://img.shields.io/github/v/release/PaoloAstrino/CUBO?color=green)](https://github.com/PaoloAstrino/CUBO/releases/latest)

- Windows: [CUBO.exe](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO.exe) (~180 MB)  
- Linux: [CUBO_linux](https://github.com/PaoloAstrino/CUBO/releases/latest/download/CUBO_linux) (PyInstaller)

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

## 💾 Efficienza della Memoria

CUBO è progettato per una **scalabilità della memoria O(1)** durante l'ingestione dei documenti. A differenza degli approcci ingenui che accumulano i chunk in RAM, CUBO utilizza:
-   **Streaming Shards:** I documenti vengono elaborati in piccoli lotti e salvati in frammenti Parquet.
-   **Pulizia Deterministica:** Trigger espliciti di garbage collection dopo ogni lotto per prevenire la frammentazione della heap.
-   **Validazione Empirica:** Testato su corpora da 0,05 GB a 1 GB (aumento di 20 volte) con un **delta di 30–44 MB** costante nell'uso della RSS.

Ciò garantisce la possibilità di ingerire corpora da oltre 50 GB su un laptop standard da 16 GB senza rallentamenti del sistema o crash.


## 🧪 Valutazione

Crediamo nel misurare, non nell'indovinare.
*   **Recall@10:** 0.96 (Politica), 0.82 (Dominio Misto).
*   **Latenza:** < 300ms per query (cache).
*   **Ingestione:** ~150 pagine/secondo.

## Benchmark Reali (embedding-gemma-300m, laptop 16 GB)

| Dataset           | Dominio       | Recall@10 | Verdetto       |
|-------------------|---------------|-----------|----------------|
| UltraDomain-Legal | Legale        | 0.48      | ⭐ Forte        |
| Politics          | Strutturato   | 0.97      | 🚀 Perfetto    |
| NFCorpus          | Medico        | 0.17      | ⚠️ Bias dominio|
| RAGBench-full     | Misto difficile| 0.30     | ⭐ Industria ok|

_Didascalia: Forte su testo legale strutturato (nostro caso d'uso principale), più debole su gergo specializzato (affrontabile con router)._ 

Vedi [examples/04_evaluation_benchmarking.ipynb](examples/04_evaluation_benchmarking.ipynb) per eseguire i tuoi benchmark.

## Per chi è CUBO?

- Avvocati / studi legali italiani che non possono caricare fascicoli su cloud (89% secondo nostro survey)
- Medici / odontoiatri con cartelle cliniche sensibili
- Ricercatori indipendenti che vogliono RAG locale senza bolletta AWS
- Chi ha solo un laptop da 16 GB e vuole privacy assoluta

## 🤝 Contribuire

I contributi sono benvenuti! Vedi [CONTRIBUTING.md](docs/CONTRIBUTING.md) per dettagli sul nostro codice di condotta e processo di sviluppo.

---

<div align="center">
  <sub>Costruito con ❤️ per Privacy ed Efficienza.</sub>
</div>
