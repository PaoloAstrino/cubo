#!/usr/bin/env python
"""Build e5-base-v2 IVFPQ index for dense retrieval baseline.

This script:
1. Loads intfloat/e5-base-v2 transformer model
2. Encodes all corpus documents to 768-dim embeddings
3. Builds FAISS IVFPQ index for fast approximate nearest neighbor search
4. Saves index to disk for query evaluation

Usage:
    python tools/index_e5_ivfpq.py \\
        --corpus-file data/beir/scifact/corpus.jsonl \\
        --output-dir data/beir_index_e5_scifact \\
        --batch-size 64
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import psutil


def load_corpus(corpus_file: Path, limit: int = None) -> Tuple[List[str], List[str]]:
    """Load corpus documents."""
    doc_ids = []
    texts = []

    print(f"Loading corpus from {corpus_file}...")

    with open(corpus_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)
            doc_id = doc.get("_id", doc.get("docid", str(i)))
            title = doc.get("title", "")
            text = doc.get("text", "")

            content = f"{title} {text}".strip()

            doc_ids.append(str(doc_id))
            texts.append(content)

            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i+1} documents...")

    print(f"[OK] Loaded {len(texts)} documents")
    return doc_ids, texts


def encode_embeddings(
    texts: List[str], batch_size: int = 64, max_memory_gb: float = 14.0
) -> np.ndarray:
    """Encode documents using e5-base-v2 model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed")
        print("  Install: pip install sentence-transformers")
        sys.exit(1)

    print(f"\nLoading e5-base-v2 model...")
    model = SentenceTransformer("intfloat/e5-base-v2")

    # Get embedding dimension
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model dimension: {dim}")

    # Encode in batches
    embeddings_list = []
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)

    print(f"\nEncoding {len(texts)} documents (batch_size={batch_size})...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)

        embeddings_list.append(batch_embeddings)

        if (i + batch_size) % (batch_size * 10) == 0:
            current_memory = process.memory_info().rss / (1024**3)
            print(f"  [{i+batch_size}/{len(texts)}] Memory: {current_memory:.1f} GB")

            if current_memory > max_memory_gb:
                print(f"[WARN] Memory usage {current_memory:.1f}GB exceeds limit {max_memory_gb}GB")

    # Combine embeddings
    embeddings = np.vstack(embeddings_list).astype(np.float32)

    final_memory = process.memory_info().rss / (1024**3)
    print(f"[OK] Encoded {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Memory used: {final_memory - initial_memory:.1f} GB")

    return embeddings


def build_ivfpq_index(
    embeddings: np.ndarray, output_dir: Path, nlist: int = 256, nbits: int = 8
) -> bool:
    """Build FAISS IVFPQ index for fast search."""

    try:
        import faiss
    except ImportError:
        print("[ERROR] faiss not installed")
        print("  Install: pip install faiss-cpu")
        sys.exit(1)

    print(f"\nBuilding IVFPQ index (nlist={nlist}, nbits={nbits})...")

    d = embeddings.shape[1]
    n = embeddings.shape[0]

    # Create IVFPQ index
    # First train IVF centroid
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, 16, nbits)

    print(f"  Training quantizer on {min(100000, n)} vectors...")
    # Train on subset if too large
    if n > 100000:
        train_indices = np.random.choice(n, 100000, replace=False)
        index.train(embeddings[train_indices])
    else:
        index.train(embeddings)

    print(f"  Adding {n} embeddings to index...")
    index.add(embeddings)

    # Save index
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "e5_ivfpq.index"
    metadata_file = output_dir / "e5_metadata.json"

    faiss.write_index(index, str(index_file))

    metadata = {
        "model": "intfloat/e5-base-v2",
        "dimension": int(d),
        "num_vectors": int(n),
        "nlist": nlist,
        "nbits": nbits,
        "index_type": "IVFPQ",
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Index saved:")
    print(f"  Index: {index_file} ({index_file.stat().st_size / (1024**2):.1f} MB)")
    print(f"  Metadata: {metadata_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build e5-base-v2 IVFPQ index")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Path to JSONL corpus")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for index")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--nlist", type=int, default=256, help="Number of IVF clusters")
    parser.add_argument("--nbits", type=int, default=8, help="Number of bits per PQ code")
    parser.add_argument("--max-memory", type=float, default=14.0, help="Max memory in GB")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only (no GPU)")

    args = parser.parse_args()

    if not args.corpus_file.exists():
        print(f"[ERROR] Corpus file not found: {args.corpus_file}")
        return 1

    print("=" * 70)
    print("E5-BASE-V2 IVFPQ INDEXING")
    print("=" * 70)

    # Load corpus
    doc_ids, texts = load_corpus(args.corpus_file)

    # Encode embeddings
    embeddings = encode_embeddings(texts, batch_size=args.batch_size, max_memory_gb=args.max_memory)

    # Save document IDs
    id_file = args.output_dir / "doc_ids.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(id_file, "w") as f:
        json.dump(doc_ids, f)

    # Build FAISS index
    success = build_ivfpq_index(embeddings, args.output_dir, args.nlist, args.nbits)

    if success:
        print(f"\n[OK] E5 index ready at: {args.output_dir}")
        return 0
    else:
        print(f"\n[ERROR] Failed to build index")
        return 1


if __name__ == "__main__":
    sys.exit(main())
