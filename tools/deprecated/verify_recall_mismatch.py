import json


def _load_valid_queries(queries_path):
    """Load valid query IDs from queries file."""
    print(f"Loading queries from {queries_path}...")
    valid_qids = set()
    with open(queries_path, "r") as f:
        for line in f:
            valid_qids.add(json.loads(line)["_id"])
    print(f"Found {len(valid_qids)} valid queries.")
    return valid_qids


def _extract_doc_ids_from_qrels_entry(q_data):
    """Extract document IDs from a qrels entry (handles list or dict format)."""
    doc_ids = set()
    if isinstance(q_data, list):
        for doc_id in q_data:
            doc_ids.add(str(doc_id))
    elif isinstance(q_data, dict):
        for doc_id in q_data.keys():
            doc_ids.add(str(doc_id))
    return doc_ids


def _load_relevant_docs(qrels_path, valid_qids):
    """Load relevant document IDs from qrels for valid queries."""
    print(f"Loading qrels from {qrels_path}...")
    with open(qrels_path, "r") as f:
        all_qrels = json.load(f)

    relevant_doc_ids = set()
    queries_with_relevant = 0

    for qid in valid_qids:
        if qid in all_qrels:
            queries_with_relevant += 1
            q_data = all_qrels[qid]
            doc_ids = _extract_doc_ids_from_qrels_entry(q_data)
            relevant_doc_ids.update(doc_ids)

    print(f"Queries with ground truth: {queries_with_relevant}")
    print(f"Total unique relevant documents required: {len(relevant_doc_ids)}")
    return relevant_doc_ids


def _scan_corpus_for_matches(corpus_path, relevant_doc_ids, limit):
    """Scan corpus and find matches with relevant documents."""
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

    return found_docs, matches


def _report_coverage_results(limit, found_docs, total_relevant):
    """Report coverage results and provide conclusion."""
    print("--- Results ---")
    print(f"Scanned Documents: {limit}")
    print(f"Relevant Documents Found: {found_docs} / {total_relevant}")
    print(f"Coverage: {(found_docs / total_relevant * 100):.2f}%")

    if found_docs == 0:
        print("\nCONCLUSION: The relevant documents are largely MISSING from the subset.")
        print("We cannot evaluate accuracy if the answers aren't in the index!")
    else:
        print("\nCONCLUSION: Some documents exist. Low recall might be a retrieval issue.")


def check_coverage(corpus_path, queries_path, qrels_path, limit=10000):
    """Check coverage of relevant documents in a limited corpus subset for recall evaluation."""
    print("--- Debugging Recall Mismatch ---")

    valid_qids = _load_valid_queries(queries_path)
    relevant_doc_ids = _load_relevant_docs(qrels_path, valid_qids)
    found_docs, matches = _scan_corpus_for_matches(corpus_path, relevant_doc_ids, limit)
    _report_coverage_results(limit, found_docs, len(relevant_doc_ids))


if __name__ == "__main__":
    # Run coverage check with default parameters
    check_coverage(
        "data/beir/beir_corpus.jsonl",
        "data/queries_valid_100.jsonl",
        "data/beir/ground_truth.json",
        limit=10000,
    )
