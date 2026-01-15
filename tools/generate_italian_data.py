import json
import os
import random

DATA_DIR = r"C:\Users\paolo\Desktop\cubo\data\italian_legal"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Generate Corpus
documents = [
    # TARGETS
    {"doc_id": "doc_01", "text": "Risoluzione del contratto. Articolo 1453 codice civile italiano. Inadempimento delle obbligazioni contrattuali."},
    {"doc_id": "doc_02", "text": "Regolamento Condominiale. Articolo dodici. Detenzione animali domestici e gatti negli spazi comuni."},
    {"doc_id": "doc_03", "text": "Diritto del lavoro. Licenziamento individuale. Il termine di preavviso obbligatorio è di trenta giorni."},
    {"doc_id": "doc_04", "text": "Sicurezza sul lavoro. I lavoratori sono tenuti all'uso dei DPI. Formazione specifica del lavoratore."},
    {"doc_id": "doc_05", "text": "Codice Civile. Diritto di proprietà. Facoltà del proprietario di godere delle cose in modo esclusivo."},
    # EASY TARGETS
    {"doc_id": "doc_06", "text": "Recesso contratto Rossi. Clausola specifica di recesso."},
]

# 200 Distractors per gap topic to bury the target unless it has the "extra" stemmed word match
for i in range(150):
    documents.append({"doc_id": f"d_cat_{i}", "text": "Regolamento Condominiale. Spazi comuni e aree condominiali. Divieto di parcheggio biciclette."})
    documents.append({"doc_id": f"d_con_{i}", "text": "Risoluzione per inadempimento. Procedura di messa in mora e risarcimento del danno."})
    documents.append({"doc_id": f"d_lic_{i}", "text": "Diritto del lavoro. Licenziamento. Periodo di preavviso secondo CCNL vigente."})
    documents.append({"doc_id": f"d_sic_{i}", "text": "Sicurezza. Formazione del personale e degli addetti ai lavori."})
    documents.append({"doc_id": f"d_pro_{i}", "text": "Codice Civile. Diritto reale. Godimento del bene e limiti di legge."})

random.seed(42)
random.shuffle(documents)

with open(os.path.join(DATA_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

# 2. 100 Queries
gap_queries = [
    {"query_id": "q_gap_1", "text": "gatto spazi comuni regolamento", "relevant_docs": ["doc_02"]},
    {"query_id": "q_gap_2", "text": "risoluzione contratti inadempimento", "relevant_docs": ["doc_01"]},
    {"query_id": "q_gap_3", "text": "termini preavviso licenziamento", "relevant_docs": ["doc_03"]},
    {"query_id": "q_gap_4", "text": "formazione sicurezza lavoratore", "relevant_docs": ["doc_04"]},
    {"query_id": "q_gap_5", "text": "proprietà godimento proprietario", "relevant_docs": ["doc_05"]}
]

easy_queries = []
for i in range(40):
    easy_queries.append({
        "query_id": f"q_easy_{i}",
        "text": "recesso contratto Rossi",
        "relevant_docs": ["doc_06"]
    })

hard_queries = []
for i in range(55):
    hard_queries.append({"query_id": f"q_hard_{i}", "text": "query vuota", "relevant_docs": ["none"]})

all_queries = gap_queries + easy_queries + hard_queries

with open(os.path.join(DATA_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
    for q in all_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Generated {len(documents)} docs and {len(all_queries)} queries.")
