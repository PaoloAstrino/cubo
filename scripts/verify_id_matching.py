import csv
import json

# 1. Load a few lines of corpus to check IDs
corpus_path = "data/beir/beir_corpus.jsonl"
print("--- Corpus IDs ---")
with open(corpus_path, "r", encoding="utf-8") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        doc = json.loads(line)
        print(f"Original ID: {doc.get('_id')}")

# 2. Check qrels IDs
qrels_path = "data/beir/qrels/dev.tsv"
print("\n--- Qrels IDs ---")
with open(qrels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for i, row in enumerate(reader):
        if i >= 5:
            break
        print(f"Qrels Query ID: {row['query-id']}, Corpus ID: {row['corpus-id']}")

# 3. Simulated retrieval check
print("\n--- Simulation ---")
test_id = "3"
# After my fix, the ID in the index will be "3" (raw string)
# In qrels, if we find "3", it's a match!
print(f"Simulated Retrieved ID: {test_id}")
match_found = False
with open(qrels_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["corpus-id"] == test_id:
            print(f"MATCH FOUND for ID {test_id} in qrels!")
            match_found = True
            break
if not match_found:
    print(f"No match for ID {test_id} in qrels (might not be in dev split).")
