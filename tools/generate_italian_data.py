import json
import os
import random

DATA_DIR = r"C:\Users\paolo\Desktop\cubo\data\italian_legal"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Generate Corpus
documents = [
    {"doc_id": "doc_00", "text": "Contratti."},
    {"doc_id": "doc_01", "text": "Gatti."},
    {"doc_id": "doc_02", "text": "Lavoratori."},
    {"doc_id": "doc_03", "text": "Proprietà."},
    {"doc_id": "doc_04", "text": "Risoluzioni."},
    {"doc_id": "doc_easy", "text": "Rossi."},
]

for i in range(150):
    documents.append({"doc_id": f"d_0_{i}", "text": "Generale."})
    documents.append({"doc_id": f"d_1_{i}", "text": "Comune."})
    documents.append({"doc_id": f"d_2_{i}", "text": "Lavoro."})
    documents.append({"doc_id": f"d_3_{i}", "text": "Civile."})
    documents.append({"doc_id": f"d_4_{i}", "text": "Penale."})

random.seed(42)
random.shuffle(documents)

with open(os.path.join(DATA_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

# 2. Queries
gap_queries = [
    {"query_id": "q_gap_0", "text": "contratto", "relevant_docs": ["doc_00"]},
    {"query_id": "q_gap_1", "text": "gatto", "relevant_docs": ["doc_01"]},
    {"query_id": "q_gap_2", "text": "lavoratore", "relevant_docs": ["doc_02"]},
    {"query_id": "q_gap_3", "text": "proprietario", "relevant_docs": ["doc_03"]},
    {"query_id": "q_gap_4", "text": "risoluzione", "relevant_docs": ["doc_04"]}
]

all_queries = gap_queries
for i in range(40):
    all_queries.append({"query_id": f"q_easy_{i}", "text": "Rossi", "relevant_docs": ["doc_easy"]})
for i in range(55):
    all_queries.append({"query_id": f"q_hard_{i}", "text": "vuota", "relevant_docs": ["none"]})

with open(os.path.join(DATA_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
    for q in all_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Generated {len(documents)} docs and {len(all_queries)} queries.")