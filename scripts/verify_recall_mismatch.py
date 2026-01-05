
import json
import os

def check_coverage(corpus_path, queries_path, qrels_path, limit=10000):
    print(f"--- Debugging Recall Mismatch ---")
    
    # 1. Load Valid Queries
    print(f"Loading queries from {queries_path}...")
    valid_qids = set()
    with open(queries_path, "r") as f:
        for line in f:
            valid_qids.add(json.loads(line)["_id"])
    print(f"Found {len(valid_qids)} valid queries.")

    # 2. Load Ground Truth for these queries
    print(f"Loading qrels from {qrels_path}...")
    with open(qrels_path, "r") as f:
        all_qrels = json.load(f)
    
    relevant_doc_ids = set()
    queries_with_relevant = 0
    for qid in valid_qids:
        if qid in all_qrels:
            queries_with_relevant += 1
            # Handle both list and dict formats for qrels
            q_data = all_qrels[qid]
            if isinstance(q_data, list):
                for doc_id in q_data:
                    relevant_doc_ids.add(str(doc_id))
            elif isinstance(q_data, dict):
                for doc_id in q_data.keys():
                    relevant_doc_ids.add(str(doc_id))
    
    print(f"Queries with ground truth: {queries_with_relevant}")
    print(f"Total unique relevant documents required: {len(relevant_doc_ids)}")

    # 3. Check presence in Corpus (simulate the limit)
    print(f"Scanning first {limit} documents in {corpus_path}...")
    
    found_docs = 0
    scanned = 0
    matches = []
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned > limit:
                break
            
            doc_id = str(json.loads(line)["_id"])
            if doc_id in relevant_doc_ids:
                found_docs += 1
                matches.append(doc_id)

    print(f"--- Results ---")
    print(f"Scanned Documents: {limit}")
    print(f"Relevant Documents Found: {found_docs} / {len(relevant_doc_ids)}")
    print(f"Coverage: {(found_docs / len(relevant_doc_ids) * 100):.2f}%")
    
    if found_docs == 0:
        print("\nCONCLUSION: The relevant documents are largely MISSING from the subset.")
        print("We cannot evaluate accuracy if the answers aren't in the index!")
    else:
        print("\nCONCLUSION: Some documents exist. Low recall might be a retrieval issue.")

if __name__ == "__main__":
    check_coverage(
        "data/beir/beir_corpus.jsonl",
        "data/queries_valid_100.jsonl",
        "data/beir/ground_truth.json",
        limit=10000
    )
