"""Generate embeddings for a Parquet file (chunks) and save as .npy

Usage:
  python tools/generate_embeddings.py --parquet data/chunks.parquet --output data/chunk_embeddings.npy --text-column text

This wrapper attempts to use the configured `EmbeddingGenerator` and falls back to a deterministic lightweight embedder to avoid heavy model download for quick workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for chunk parquet and save as .npy"
    )
    parser.add_argument("--parquet", required=True, help="Path to parquet file with chunk text")
    parser.add_argument("--output", required=True, help="Path to .npy output")
    parser.add_argument("--text-column", default="text", help="Text column to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    return parser.parse_args()


def _lightweight_embedder(texts):
    # A deterministic but cheap embedding based on token counts and length
    out = []
    for t in texts:
        s = str(t)
        out.append([len(s), s.count(" "), s.count(","), s.count(".")])
    return np.array(out, dtype=np.float32)


def main():
    args = parse_args()
    parquet = Path(args.parquet)
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")
    df = pd.read_parquet(parquet)
    texts = df[args.text_column].fillna("").astype(str).tolist()

    # Lazy import to avoid heavy dependencies if running in lightweight environments
    try:
        from cubo.embeddings.embedding_generator import EmbeddingGenerator

        logger.info("Using EmbeddingGenerator")
        gen = EmbeddingGenerator(batch_size=args.batch_size)
        embeddings = gen.encode(texts, batch_size=args.batch_size)
    except Exception as e:
        logger.warning(
            "EmbeddingGenerator unavailable or load failed, falling back to lightweight embedder: %s",
            e,
        )
        embeddings = _lightweight_embedder(texts)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), np.asarray(embeddings, dtype=np.float32))
    logger.info(f"Wrote embeddings to {output} (shape: {np.asarray(embeddings).shape})")


if __name__ == "__main__":
    main()
