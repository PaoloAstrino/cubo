import json
import random


def create_smart_corpus(
    corpus_input="data/beir/beir_corpus.jsonl",
    qrels_input="data/beir/ground_truth.json",
    queries_input="data/queries_valid_100.jsonl",
    output_path="data/beir/beir_corpus_smart.jsonl",
    target_size=10000,
):
    print("--- Creating Smart Benchmark Corpus ---")

    # 1. Inspect valid queries to get relevant doc IDs
    print(f"Loading queries from {queries_input}...")
    valid_qids = set()
    with open(queries_input, "r") as f:
        for line in f:
            valid_qids.add(json.loads(line)["_id"])

    # 2. Extract needed Document IDs
    print(f"Loading qrels from {qrels_input}...")
    with open(qrels_input, "r") as f:
        all_qrels = json.load(f)

    needed_doc_ids = set()
    for qid in valid_qids:
        if qid in all_qrels:
            q_data = all_qrels[qid]
            if isinstance(q_data, list):
                for did in q_data:
                    needed_doc_ids.add(str(did))
            elif isinstance(q_data, dict):
                for did in q_data.keys():
                    needed_doc_ids.add(str(did))

    print(f"Relevant documents to rescue: {len(needed_doc_ids)}")

    # 3. Scan Corpus and Rescue Documents
    print("Scanning full corpus to extract relevant docs + distractors...")

    rescued_docs = []
    distractor_candidates = []

    with open(corpus_input, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            did = str(doc["_id"])

            if did in needed_doc_ids:
                rescued_docs.append(line)
                needed_doc_ids.remove(did)
            else:
                # Reservoir sampling for distractors if we wanted to be fancy,
                # but simple append and slice is fine for 57k docs -> 10k target
                if len(distractor_candidates) < (target_size * 2):  # collect a few more
                    distractor_candidates.append(line)

    print(f"Rescued {len(rescued_docs)} relevant documents.")
    if needed_doc_ids:
        print(f"WARNING: {len(needed_doc_ids)} relevant documents were NOT FOUND in the corpus!")

    # 4. Fill the rest with distractors
    remaining_slots = target_size - len(rescued_docs)
    selected_distractors = distractor_candidates[:remaining_slots]

    final_corpus = rescued_docs + selected_distractors

    # Shuffle to simulate real distribution (optional, but good practice)
    random.shuffle(final_corpus)

    print(f"Writing {len(final_corpus)} documents to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in final_corpus:
            f.write(line)

    print("--- Smart Corpus Created ---")


if __name__ == "__main__":
    create_smart_corpus()
