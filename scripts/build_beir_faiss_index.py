"""Build FAISS index from BEIR corpus.jsonl file.

This script:
1. Reads the BEIR corpus from data/beir/corpus.jsonl
2. Generates embeddings using the configured model
3. Builds FAISS hot/cold indexes
4. Saves the index and documents.db
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from BEIR corpus")
    parser.add_argument(
        "--corpus",
        default="data/beir/corpus.jsonl",
        help="Path to corpus.jsonl file",
    )
    parser.add_argument(
        "--index-dir",
        default=config.get("faiss_index_dir", "faiss_index"),
        help="Directory to save FAISS index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get("embedding_batch_size", 32),
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--hot-fraction",
        type=float,
        default=0.25,
        help="Fraction of documents to keep in hot index",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=64,
        help="Number of clusters for IVF index",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=16,
        help="HNSW M parameter",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize embeddings to unit vectors (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save the index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)",
    )
    return parser.parse_args()


def load_corpus(corpus_path: str, limit: int = None) -> Tuple[List[str], List[str], List[str]]:
    """Load documents from BEIR corpus.jsonl.
    
    Returns:
        Tuple of (doc_ids, texts, titles)
    """
    doc_ids = []
    texts = []
    titles = []
    
    logger.info(f"Loading corpus from {corpus_path}...")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            doc = json.loads(line)
            doc_ids.append(str(doc["_id"]))
            texts.append(doc.get("text", ""))
            titles.append(doc.get("title", ""))
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Loaded {i + 1} documents...")
    
    logger.info(f"Loaded {len(doc_ids)} documents from corpus")
    return doc_ids, texts, titles


def create_documents_db(
    index_dir: Path,
    doc_ids: List[str],
    texts: List[str],
    titles: List[str],
    corpus_path: str,
) -> None:
    """Create documents.db SQLite database."""
    db_path = index_dir / "documents.db"
    
    # Remove existing DB
    if db_path.exists():
        db_path.unlink()
    
    logger.info(f"Creating documents database at {db_path}...")
    
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    
    # Create table with same schema as expected
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL
        )
    """)
    
    # Insert documents
    for doc_id, text, title in zip(doc_ids, texts, titles):
        metadata = json.dumps({
            "filename": "corpus.jsonl",
            "file_path": corpus_path,
            "title": title,
        })
        cur.execute(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
            (doc_id, text, metadata),
        )
    
    conn.commit()
    logger.info(f"Inserted {len(doc_ids)} documents into database")
    
    # Verify
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    logger.info(f"Verified: {count} documents in database")
    
    conn.close()


def main():
    args = parse_args()
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error(f"Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    index_dir = Path(args.index_dir)
    
    # CRITICAL: Delete existing index completely to prevent stale FAISS index files
    if index_dir.exists():
        logger.info(f"Removing existing index at {index_dir}")
        import shutil
        shutil.rmtree(index_dir, ignore_errors=True)
    
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    doc_ids, texts, titles = load_corpus(str(corpus_path), limit=args.limit)
    
    if not texts:
        logger.error("No documents loaded from corpus")
        sys.exit(1)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    generator = EmbeddingGenerator(batch_size=args.batch_size)
    embeddings = generator.encode(texts, batch_size=args.batch_size)
    
    dimension = len(embeddings[0]) if embeddings else 0
    if dimension == 0:
        logger.error("Unable to determine embedding dimension")
        sys.exit(1)
    
    logger.info(f"Generated {len(embeddings)} embeddings with dimension {dimension}")
    
    # Get model path for metadata
    model_path = getattr(generator, 'model_path', None) or config.get('model_path')
    
    # Build FAISS index
    logger.info(f"Building FAISS index (normalize={args.normalize})...")
    manager = FAISSIndexManager(
        dimension=dimension,
        index_dir=index_dir,
        nlist=args.nlist,
        hnsw_m=args.hnsw_m,
        hot_fraction=args.hot_fraction,
        normalize=args.normalize,
        model_path=model_path,
    )
    manager.build_indexes(embeddings, doc_ids)
    
    # Sample search to verify
    sample_query = embeddings[0]
    hits = manager.search(sample_query, k=min(5, len(doc_ids)))
    logger.info(f"Sample search returned {len(hits)} hits")
    if hits:
        logger.info(f"First hit: ID={hits[0]['id']}, score={hits[0].get('score', 'N/A')}")
    
    if not args.dry_run:
        # Save FAISS index
        manager.save()
        logger.info(f"FAISS index saved to {index_dir}")
        
        # Create documents.db
        create_documents_db(index_dir, doc_ids, texts, titles, str(corpus_path))
        
        logger.info("BEIR FAISS index build complete!")
    else:
        logger.info("Dry-run mode - index not saved")


if __name__ == "__main__":
    main()
