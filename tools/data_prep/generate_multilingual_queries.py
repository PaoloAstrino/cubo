import json
import os
import random


def generate_language_data(lang_code, lang_name, data_map):
    DATA_DIR = rf"C:\Users\paolo\Desktop\cubo\data\legal_{lang_code}"
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Generate Corpus
    documents = []
    # TARGETS (Stemming gap)
    for i, target in enumerate(data_map["targets"]):
        documents.append({"doc_id": f"doc_{i:02d}", "text": target})

    # EASY TARGET
    documents.append({"doc_id": "doc_easy", "text": data_map["easy_target"]})

    # Distractors
    for i in range(150):
        for j, distractor in enumerate(data_map["distractors"]):
            documents.append({"doc_id": f"d_{j}_{i}", "text": distractor})

    random.seed(42)
    random.shuffle(documents)

    with open(os.path.join(DATA_DIR, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 2. Queries
    all_queries = []
    # Gap Queries (benefit from stemming)
    # Use ONLY the word that needs stemming
    for i, q_text in enumerate(data_map["gap_queries"]):
        all_queries.append(
            {"query_id": f"q_gap_{i}", "text": q_text, "relevant_docs": [f"doc_{i:02d}"]}
        )

    # Easy Queries (exact match)
    for i in range(40):
        all_queries.append(
            {
                "query_id": f"q_easy_{i}",
                "text": data_map["easy_query"],
                "relevant_docs": ["doc_easy"],
            }
        )

    # Hard Queries (noise)
    for i in range(55):
        all_queries.append(
            {"query_id": f"q_hard_{i}", "text": data_map["hard_query"], "relevant_docs": ["none"]}
        )

    with open(os.path.join(DATA_DIR, "queries.jsonl"), "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(
        f"Generated {len(documents)} docs and {len(all_queries)} queries for {lang_name} ({lang_code})."
    )


# Data definitions
data_configs = {
    "fr": {
        "targets": [
            "Réglementation des contrats.",
            "Les résiliations de bail.",
            "La responsabilité des entreprises.",
            "Les cotisations sociales.",
            "Protection des brevets.",
        ],
        "gap_queries": ["contrat", "résiliation", "entreprise", "cotisation", "brevet"],
        "easy_target": "Maintenance Dupont.",
        "easy_query": "Maintenance Dupont",
        "distractors": ["Général.", "Administratif.", "Fiscal.", "Famille.", "Pénal."],
        "hard_query": "vide",
    },
    "de": {
        "targets": ["Arbeitsverträge.", "Mietverträge.", "Schäden.", "Beiträge.", "Patenten."],
        "gap_queries": ["Arbeitsvertrag", "Mietvertrag", "Schaden", "Beitrag", "Patent"],
        "easy_target": "Wartung Müller.",
        "easy_query": "Wartung Müller",
        "distractors": ["Allgemein.", "Verwaltung.", "Steuer.", "Familie.", "Straf."],
        "hard_query": "leer",
    },
    "es": {
        "targets": ["Contratos.", "Rescisiones.", "Empresas.", "Cotizaciones.", "Patentes."],
        "gap_queries": ["contrato", "rescisión", "empresa", "cotización", "patente"],
        "easy_target": "Mantenimiento García.",
        "easy_query": "Mantenimiento García",
        "distractors": ["General.", "Administrativo.", "Fiscal.", "Familia.", "Penal."],
        "hard_query": "vacío",
    },
}

if __name__ == "__main__":
    for lang, config in data_configs.items():
        generate_language_data(lang, lang.upper(), config)
