# CUBO - Assistente Documentale AI v1.3.0

**Nota (2025-12-17):** Pulizia della documentazione — versione allineata, esempi `src`→`cubo` corretti, rimosse sezioni dipendenze duplicate e aggiornati i flag di coverage di pytest.

🌍 **[English](README.md)** | **Italiano** | **[中文](README.zh.md)**

[![CI/CD](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/cubo/actions/workflows/ci-cd.yml)
[![E2E Tests](https://github.com/your-username/cubo/actions/workflows/e2e.yml/badge.svg)](https://github.com/your-username/cubo/actions/workflows/e2e.yml)

Un sistema RAG (Retrieval-Augmented Generation) modulare che utilizza modelli di embedding e Large Language Models (LLM) con un'interfaccia desktop moderna e API web. **Progettato per la privacy**: funziona completamente offline, i tuoi dati non lasciano mai il tuo computer.

## 🌟 Perché CUBO?

- **🔒 Privacy First**: 100% offline - nessun dato inviato a server esterni
- **💻 Laptop-Friendly**: Ottimizzato per laptop con risorse limitate (≤16GB RAM)
- **🇪🇺 GDPR Compliant**: Audit log esportabili, cancellazione documenti, scrubbing query
- **🚀 Facile da Usare**: Nessuna configurazione tecnica richiesta

## Novità in v1.3.1

- **📝 API Citazioni**: Ogni risposta include citazioni GDPR-compliant con file sorgente, pagina, chunk_id
- **🗑️ Cancellazione Documenti**: Endpoint DELETE per rimuovere documenti dall'indice (GDPR Art. 17)
- **📊 Export Audit GDPR**: Esporta log in CSV/JSON per audit di conformità
- **🌍 README Multilingue**: Documentazione in Italiano, Inglese e Cinese

## Avvio Rapido

### Interfaccia Web (Consigliata)

```bash
# Installa dipendenze
pip install -r requirements.txt
cd frontend && pnpm install && cd ..

# Avvia backend e frontend
python scripts/start_fullstack.py
```

Visita:
- Frontend: http://localhost:3000
- Documentazione API: http://localhost:8000/docs

### Interfaccia Desktop

```bash
pip install -r requirements.txt
python launch_gui.py
```

## API Endpoints

### Query con Citazioni

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Cosa dice il contratto sulla garanzia?", "top_k": 5}'
```

**Risposta:**
```json
{
  "answer": "Il contratto prevede una garanzia di 24 mesi...",
  "sources": [...],
  "citations": [
    {
      "source_file": "contratto_2024.pdf",
      "page": 5,
      "chunk_id": "abc123",
      "chunk_index": 12,
      "text_snippet": "La garanzia ha durata di 24 mesi dalla consegna...",
      "relevance_score": 0.95
    }
  ],
  "trace_id": "tr_abc123",
  "query_scrubbed": false
}
```

### Cancellazione Documento (GDPR Art. 17)

```bash
curl -X DELETE http://localhost:8000/api/documents/contratto_2024.pdf
```

**Risposta:**
```json
{
  "doc_id": "contratto_2024.pdf",
  "deleted": true,
  "chunks_removed": 15,
  "trace_id": "tr_xyz789",
  "message": "Document contratto_2024.pdf deleted successfully"
}
```

### Export Audit GDPR

```bash
# Esporta audit come CSV
curl "http://localhost:8000/api/export-audit?start_date=2024-11-01&format=csv" > audit.csv

# Esporta come JSON
curl "http://localhost:8000/api/export-audit?format=json" > audit.json
```

## Funzionalità

- **🔍 Ricerca Ibrida**: Combina BM25 (keyword) e FAISS (semantica) per risultati precisi
- **🐬 Elaborazione Avanzata (opzionale)**: Supporto per modelli vision-language esterni (es. Dolphin) per parsing PDF/immagini migliorato (opzionale).
- **📊 Deduplicazione Semantica**: Riduce chunk duplicati con MinHash + FAISS + HDBSCAN
- **🔄 Reranker**: Cross-encoder per migliorare l'ordinamento dei risultati
- **📝 Sentence Window**: Chunking intelligente con finestre di contesto configurabili
- **🔒 Scrubbing Query**: Anonimizzazione automatica delle query nei log

## Modalità Laptop

CUBO rileva automaticamente hardware laptop (≤16GB RAM o ≤6 core CPU) e abilita ottimizzazioni:

```bash
# Forza modalità laptop
export CUBO_LAPTOP_MODE=1
python start_api_server.py --mode laptop
```

## Requisiti

- Python 3.8+
- Ollama (per generazione LLM)
- GPU CUDA-compatibile (opzionale)

### Installazione Rapida

```bash
# Crea ambiente virtuale
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installa dipendenze
pip install -r requirements.txt

# Avvia Ollama e scarica modello
ollama pull llama3.2:latest
```

## Configurazione

Modifica `config.json`:

```json
{
  "model_path": "./models/all-MiniLM-L6-v2",
  "llm_model": "llama3.2:latest",
  "top_k": 5,
  "chunk_size": 512
}
```

### Variabili d'Ambiente

```bash
export CUBO_ENCRYPTION_KEY="your-key"
export CUBO_LAPTOP_MODE=1
```

## Casi d'Uso Enterprise

### Legale e Compliance
- Analisi contratti e clausole
- Ricerca documenti normativi
- Due diligence

### Sanità e Ricerca
- Interrogazione cartelle cliniche (HIPAA-compliant)
- Revisione letteratura scientifica
- Linee guida cliniche

### Documentazione Tecnica
- Knowledge base aziendale
- Documentazione API
- Troubleshooting

## Test

```bash
# Esegui tutti i test
python -m pytest -q

# Test con coverage
python -m pytest --cov=cubo --cov-report=html
```

## Sicurezza

- **Sanitizzazione Path**: Previene attacchi directory traversal
- **Limiti File**: Dimensione massima configurabile
- **Rate Limiting**: Previene abusi
- **Crittografia**: Dati sensibili crittografati con Fernet
- **Audit Logging**: Azioni logate per compliance

## Licenza

MIT License - vedi [LICENSE](LICENSE)

## Contribuire

Vedi [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.

## Supporto

- 📖 [Documentazione API](docs/API_INTEGRATION.md)
- 🐛 [Segnala Bug](https://github.com/your-username/cubo/issues)
- 💬 [Discussioni](https://github.com/your-username/cubo/discussions)
