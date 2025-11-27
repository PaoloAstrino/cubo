#!/usr/bin/env python3
"""
Prepare BEIR corpus for CUBO testing by preserving document IDs.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_beir_corpus():
    """Load BEIR corpus and prepare it with preserved IDs."""
    print("Loading BEIR corpus...")

    # Load BEIR corpus with IDs preserved
    corpus_docs = []
    with open("data/beir/corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line.strip())
                corpus_docs.append(doc)

    print(f"Loaded {len(corpus_docs)} corpus documents")
    print(f'First doc ID: {corpus_docs[0]["_id"]}')
    print(f'First doc text preview: {corpus_docs[0]["text"][:100]}...')

    # Save as a format the system can use
    output_docs = []
    for doc in corpus_docs:
        # Create a document chunk with preserved ID
        output_docs.append(
            {
                "id": doc["_id"],
                "text": doc["text"],
                "title": doc.get("title", ""),
                "chunk_index": 0,
                "filename": f'beir_corpus_{doc["_id"]}.txt',
                "file_path": f'data/beir/corpus_{doc["_id"]}.txt',
            }
        )

    with open("data/beir/corpus_processed.json", "w", encoding="utf-8") as f:
        json.dump(output_docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(output_docs)} processed documents to corpus_processed.json")
    return output_docs


if __name__ == "__main__":
    prepare_beir_corpus()
