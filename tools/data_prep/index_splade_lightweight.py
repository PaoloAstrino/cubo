#!/usr/bin/env python
"""Build SPLADE sparse vector index for retrieval.

Uses naver/splade-cocondenser-ensembledistil model to encode documents
as sparse vectors, storing in lightweight format with ID mapping.

SPLADE encodes documents as sparse term vectors (similar to BM25 but learned).
This tool does NOT require GPU - CPU-only inference.

Usage:
    python tools/index_splade_lightweight.py \\
        --corpus-file data/beir/scifact/corpus.jsonl \\
        --output-dir data/beir_index_splade_scifact \\
        --batch-size 64
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil


def load_corpus(corpus_file: Path, limit: int = None) -> tuple:
    """Load documents from BEIR corpus JSONL."""
    docs = {}
    doc_ids = []

    with open(corpus_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            doc_id = doc.get("_id", doc.get("id", str(i)))
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            docs[doc_id] = text
            doc_ids.append(doc_id)

    return docs, doc_ids


def encode_splade(
    texts: List[str], model_name: str = "naver/splade-cocondenser-ensembledistil"
) -> np.ndarray:
    """Encode texts as SPLADE sparse vectors."""

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("[HINT] Install with: pip install torch transformers")
        sys.exit(1)

    print(f"Loading SPLADE model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=".cache")

    # Move to CPU (set device appropriately for GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Using device: {device}")
    model = model.to(device)
    model.eval()

    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Compute sparse representations
    # Take max over sequence dimension, apply ReLU
    sparse_vecs = torch.max(torch.zeros_like(logits), logits)  # [batch, seq_len, vocab_size]
    sparse_vecs = torch.amax(sparse_vecs, dim=1)  # [batch, vocab_size]

    return sparse_vecs.cpu().numpy()


def build_sparse_index(docs: Dict, doc_ids: List, batch_size: int = 32) -> tuple:
    """Build sparse vector index from documents."""

    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Missing SPLADE dependencies: {e}")
        sys.exit(1)

    print(f"Loading SPLADE model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "naver/splade-cocondenser-ensembledistil", cache_dir=".cache"
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "naver/splade-cocondenser-ensembledistil", cache_dir=".cache"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"\nEncoding {len(docs)} documents as SPLADE sparse vectors (batch_size={batch_size})...")

    # Store sparse vectors as dict: doc_id -> {term_id: weight}
    sparse_vectors = {}
    vocab_size = tokenizer.vocab_size

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)
    peak_memory = initial_memory

    start_time = time.time()

    for batch_start in range(0, len(doc_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(doc_ids))
        batch_ids = doc_ids[batch_start:batch_end]
        batch_texts = [docs[doc_id] for doc_id in batch_ids]

        # Tokenize
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Compute sparse representations: max over sequence dimension
        sparse_vecs = torch.relu(logits)  # Apply ReLU
        sparse_vecs = torch.amax(sparse_vecs, dim=1)  # [batch, vocab_size]

        # Convert to dict format (keep only non-zero weights)
        batch_sparse = sparse_vecs.cpu().numpy()

        for doc_id, sparse_vec in zip(batch_ids, batch_sparse):
            # Keep only non-zero indices with weights
            nonzero_indices = np.nonzero(sparse_vec)[0]
            if len(nonzero_indices) > 0:
                # Store as {term_id: weight}
                sparse_vectors[doc_id] = {
                    str(int(idx)): float(sparse_vec[idx]) for idx in nonzero_indices
                }
            else:
                sparse_vectors[doc_id] = {}

        # Track memory
        current_memory = process.memory_info().rss / (1024**3)
        peak_memory = max(peak_memory, current_memory)

        elapsed = time.time() - start_time
        rate = (batch_end / elapsed) if elapsed > 0 else 0
        eta = (len(doc_ids) - batch_end) / rate if rate > 0 else 0

        if batch_end % max(100, len(doc_ids) // 10) == 0 or batch_end == len(doc_ids):
            print(
                f"  [{batch_end}/{len(doc_ids)}] Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s, Memory: {peak_memory:.1f}GB"
            )

    total_time = time.time() - start_time
    print(
        f"\n[OK] Encoded {len(docs)} documents in {total_time:.0f}s ({len(docs)/total_time:.1f} docs/s)"
    )
    print(f"     Memory peak: {peak_memory:.1f}GB")

    return sparse_vectors, vocab_size


def main():
    parser = argparse.ArgumentParser(description="Build SPLADE sparse vector index")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Corpus JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for index")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit documents (for testing)")

    args = parser.parse_args()

    if not args.corpus_file.exists():
        print(f"[ERROR] Corpus file not found: {args.corpus_file}")
        return 1

    print("=" * 70)
    print("SPLADE SPARSE VECTOR INDEXING")
    print("=" * 70)

    # Load corpus
    print(f"Loading corpus from {args.corpus_file}...")
    docs, doc_ids = load_corpus(args.corpus_file, args.limit)
    print(f"[OK] Loaded {len(docs)} documents")

    # Build index
    sparse_vectors, vocab_size = build_sparse_index(docs, doc_ids, args.batch_size)

    # Save index
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save sparse vectors as JSON
    index_file = args.output_dir / "splade_index.json"
    print(f"\nSaving sparse vectors to {index_file}...")
    with open(index_file, "w") as f:
        json.dump(sparse_vectors, f)  # No indentation to save space

    # Save doc_ids
    doc_ids_file = args.output_dir / "doc_ids.json"
    with open(doc_ids_file, "w") as f:
        json.dump(doc_ids, f)

    # Save metadata
    metadata = {
        "num_documents": len(docs),
        "vocab_size": vocab_size,
        "model": "naver/splade-cocondenser-ensembledistil",
    }
    metadata_file = args.output_dir / "splade_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    index_size_mb = index_file.stat().st_size / (1024**2) if index_file.exists() else 0

    print(f"\n{'=' * 70}")
    print(f"[OK] Index saved to {args.output_dir}")
    print(f"     Vectors: {index_file} ({index_size_mb:.1f} MB)")
    print(f"     Doc IDs: {doc_ids_file}")
    print(f"     Metadata: {metadata_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
