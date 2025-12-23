import json
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cubo.config import config
from cubo.config.settings import settings
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.retrieval.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class CuboBeirAdapter:
    """
    Adapter to run BEIR benchmarks using Cubo's actual DocumentRetriever.
    This ensures metrics reflect the full production pipeline (Hybrid Search, Reranking, etc).
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        lightweight: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            index_dir: Directory containing FAISS index and documents.db
            embedding_generator: Pre-initialized embedding generator (optional)
            lightweight: If True, avoids loading heavy models (useful for unit tests only)
        """
        self.index_dir = index_dir
        self.embedding_generator = embedding_generator
        self.retriever = None
        self.lightweight = lightweight

        # If not lightweight, initialize the full retriever stack
        if not self.lightweight:
            self._initialize_retriever()

    def _initialize_retriever(self):
        """Initialize the full DocumentRetriever stack."""
        if self.retriever:
            return

        logger.info("Initializing DocumentRetriever stack...")

        # 1. Load Model (if not provided)
        if not self.embedding_generator:
            self.embedding_generator = EmbeddingGenerator()

        # 2. Initialize Retriever
        # We use the settings from config/settings.py to match production
        self.retriever = DocumentRetriever(
            model=self.embedding_generator.model,  # Pass the underlying SentenceTransformer
            top_k=settings.retrieval.default_top_k,
            window_size=settings.retrieval.default_window_size,
            use_sentence_window=config.get("chunking.use_sentence_window", True),
            use_reranker=config.get("retrieval.use_reranker", True),
            # Disable auto-merging for BEIR as it expects flat passages usually
            use_auto_merging=False,
        )

        # 3. Point retriever to the specific index_dir if provided
        # DocumentRetriever usually loads from config paths.
        # We need to ensure it sees our specific index_dir if we are benchmarking a custom build.
        # The DocumentStore/FaissStore inside retriever loads from config.
        # For this adapter to work with a custom dir, we might need to patch the store
        # or ensure config points to it.
        # For now, we assume the user configured the app to point to index_dir OR
        # we manually reload the store.
        if self.index_dir:
            logger.info(f"Pointing retriever to {self.index_dir}")
            # We can manually reload the vector store with the new path
            # This depends on FaissStore implementation exposing a load method or similar
            # Or we just rely on the fact that we set the config before init?
            # Let's assume for now we are running in a process where we can't easily change global config
            # without side effects.
            # Ideally, DocumentRetriever should accept a path.
            # Since it doesn't, we might need to hack it or rely on standard paths.
            # BUT: FaissStore has a load() method.
            if hasattr(self.retriever, "collection"):
                # Update index_dir and db_path to point to the benchmark index
                if self.index_dir:
                    new_path = Path(self.index_dir)
                    if hasattr(self.retriever.collection, "index_dir"):
                        self.retriever.collection.index_dir = new_path
                    if hasattr(self.retriever.collection, "_db_path"):
                        self.retriever.collection._db_path = new_path / "documents.db"

                    # Reload if possible
                    if (
                        hasattr(self.retriever.collection, "load")
                        and (new_path / "metadata.json").exists()
                    ):
                        self.retriever.collection.load(new_path)
                    else:
                        logger.warning(
                            f"Index directory {self.index_dir} exists but no metadata.json found. Skipping load."
                        )

    def index_corpus(
        self,
        corpus_path: str,
        index_dir: str,
        batch_size: int = 512,
        limit: Optional[int] = None,
        normalize: bool = True,
    ) -> int:
        """
        Index a BEIR corpus (JSONL) into FAISS and SQLite.

        Args:
            corpus_path: Path to corpus.jsonl
            index_dir: Output directory
            batch_size: Batch size for embedding
            limit: Max documents to index (for testing)
            normalize: Whether to normalize vectors

        Returns:
            Number of documents indexed
        """
        # For indexing, we can still use the direct FAISS approach for speed,
        # OR we can use the retriever's add_documents method.
        # Using direct FAISS approach is faster for bulk and standard for benchmarks.
        # The key is that the resulting index structure MUST match what DocumentRetriever expects.
        # DocumentRetriever expects:
        # - SQLite 'documents' table with 'id', 'content', 'metadata'
        # - FAISS index file

        # We will reuse the previous implementation for indexing as it produces compatible structures.
        # Just ensure we initialize the generator if needed.
        if not self.embedding_generator:
            self.embedding_generator = EmbeddingGenerator()
            self.embedding_generator._load_model()

        # ... (Reuse the efficient indexing logic from before) ...
        # COPYING PREVIOUS LOGIC FOR COMPLETENESS

        index_path = Path(index_dir)
        if index_path.exists():
            logger.warning(f"Removing existing index directory: {index_dir}")
            shutil.rmtree(index_dir, ignore_errors=True)
        index_path.mkdir(parents=True, exist_ok=True)

        faiss_manager = FAISSIndexManager(
            dimension=self.embedding_generator.model.get_sentence_embedding_dimension(),
            index_dir=index_path,
            # Use 100% hot index (HNSW) for benchmarks - no IVF clustering needed
            hot_fraction=1.0,
            # Smaller nlist as fallback if cold index is ever used
            nlist=16,
        )

        db_path = index_path / "documents.db"
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT, metadata TEXT)"
        )
        # Also create collections table as DocumentStore might check it
        c.execute(
            "CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, color TEXT)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector BLOB, dtype TEXT, dim INTEGER, created_at TEXT)"
        )

        count = 0
        batch_docs = []
        batch_ids = []

        logger.info(f"Indexing corpus from {corpus_path}...")

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit and count >= limit:
                    break
                try:
                    doc = json.loads(line)
                    doc_id = str(doc.get("_id"))
                    text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    if not text:
                        continue
                    batch_docs.append(text)
                    batch_ids.append(doc_id)
                    if len(batch_docs) >= batch_size:
                        self._process_batch(batch_docs, batch_ids, conn, normalize, faiss_manager)
                        count += len(batch_docs)
                        batch_docs = []
                        batch_ids = []
                        if count % 10000 == 0:
                            logger.info(f"Indexed {count} documents...")
                except json.JSONDecodeError:
                    continue

        if batch_docs:
            self._process_batch(batch_docs, batch_ids, conn, normalize, faiss_manager)
            count += len(batch_docs)

        logger.info("Saving FAISS index...")
        faiss_manager.save(index_path)
        conn.commit()
        conn.close()

        self.index_dir = index_dir
        # After indexing, if we have a retriever, reload it
        if self.retriever and hasattr(self.retriever, "collection"):
            self.retriever.collection.load(self.index_dir)

        logger.info(f"Indexing complete. {count} documents indexed.")
        return count

    def _process_batch(self, texts, ids, conn, normalize, faiss_manager):
        """Helper to embed and add batch to FAISS and DB."""
        # Use direct model.encode() for bulk indexing to avoid threading timeout issues
        # Use smaller internal batch size to avoid GPU OOM
        try:
            embeddings = self.embedding_generator.model.encode(
                texts,
                batch_size=8,  # Small batch to avoid GPU OOM
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:
            # Fallback for models that don't support all parameters
            embeddings = self.embedding_generator.model.encode(texts, batch_size=8)

        # Convert to numpy if needed
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        # Use build_indexes with append=True to add to existing index
        faiss_manager.build_indexes(embeddings, ids, append=True)

        c = conn.cursor()
        data = []
        for doc_id, text in zip(ids, texts):
            # We store metadata as JSON string
            data.append((doc_id, text, json.dumps({"id": doc_id, "source": "beir"})))
        c.executemany(
            "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)", data
        )
        conn.commit()

    def load_index(self, index_dir: str):
        """Load existing index."""
        self.index_dir = index_dir
        if not self.lightweight:
            self._initialize_retriever()
            # Force reload from specific dir
            if hasattr(self.retriever, "collection") and hasattr(self.retriever.collection, "load"):
                self.retriever.collection.load(self.index_dir)

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve documents for a single query using full DocumentRetriever."""
        if not self.retriever:
            raise ValueError(
                "Retriever not initialized. Call load_index() or init with lightweight=False."
            )

        # Use the full production pipeline
        # retrieve_top_documents returns List[Dict] with 'id', 'similarity', 'document', etc.
        results = self.retriever.retrieve_top_documents(query, top_k=top_k)

        # Convert to (doc_id, score) tuples for BEIR
        return [(r["id"], r["similarity"]) for r in results]

    def retrieve_bulk_optimized(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        skip_reranker: bool = True,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimized bulk retrieval for benchmarking.

        This method is significantly faster than retrieve_bulk() because it:
        1. Batches query embeddings (6000+ queries in ~30s vs 6000+ individual calls)
        2. Optionally skips reranking (saves ~8s per query)
        3. Uses direct FAISS batch search instead of the full production pipeline

        Args:
            queries: Dict mapping query_id -> query_text
            top_k: Number of results per query
            skip_reranker: If True, skip reranking (much faster, recommended for benchmarking)
            batch_size: Batch size for embedding generation

        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized.")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        total = len(query_ids)

        logger.info(f"Optimized retrieval for {total} queries (skip_reranker={skip_reranker})...")

        # Step 1: Batch generate embeddings for all queries
        logger.info("Generating query embeddings in batches...")
        query_embeddings = []
        for i in range(0, len(query_texts), batch_size):
            batch = query_texts[i : i + batch_size]
            try:
                batch_embs = self.embedding_generator.model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                if hasattr(batch_embs, "cpu"):
                    batch_embs = batch_embs.cpu().numpy()
                query_embeddings.extend(batch_embs)

                if (i + len(batch)) % 1000 == 0:
                    logger.info(f"Embedded {i + len(batch)}/{total} queries")
            except Exception as e:
                logger.error(f"Error embedding batch {i}: {e}")
                # Fallback to individual embeddings for this batch
                for q in batch:
                    try:
                        emb = self.embedding_generator.model.encode([q], convert_to_numpy=True)
                        if hasattr(emb, "cpu"):
                            emb = emb.cpu().numpy()
                        query_embeddings.append(emb[0])
                    except Exception:
                        # Use zero vector as last resort
                        query_embeddings.append(
                            np.zeros(
                                self.embedding_generator.model.get_sentence_embedding_dimension()
                            )
                        )

        query_embeddings = np.array(query_embeddings)
        logger.info(f"Generated {len(query_embeddings)} query embeddings")

        # Step 2: Batch FAISS search
        logger.info("Performing batch FAISS search...")
        results = {}

        # Load FAISS index directly from disk
        try:
            import faiss

            index_path = Path(self.index_dir) / "hot.index"
            if not index_path.exists():
                logger.error(f"FAISS index not found at {index_path}")
                return {qid: {} for qid in query_ids}

            logger.info(f"Loading FAISS index from {index_path}")
            faiss_index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")

            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            query_embeddings = query_embeddings / (norms + 1e-10)

            # Batch search
            distances, indices = faiss_index.search(query_embeddings, top_k)

            # Convert FAISS results to BEIR format
            # Get document IDs from the metadata.json which contains the index-to-ID mapping
            metadata_path = Path(self.index_dir) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                # Use hot_ids since we're using hot index (100% hot fraction for benchmarks)
                all_doc_ids = metadata.get("hot_ids", [])
                if not all_doc_ids:
                    # Fallback to cold_ids if hot_ids is empty
                    all_doc_ids = metadata.get("cold_ids", [])
                logger.info(f"Loaded {len(all_doc_ids)} document IDs from metadata.json")
            else:
                # Fallback to database query
                logger.warning("metadata.json not found, falling back to database query")
                # Use the retriever's collection if available
                coll = getattr(self.retriever, "collection", None)
                if coll is None:
                    logger.error("No collection available on retriever to extract document IDs")
                    all_doc_ids = []
                else:
                    if hasattr(coll, "_db_path"):
                        # SQLite-backed collection
                        import sqlite3

                        conn = sqlite3.connect(str(coll._db_path))
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM documents")
                        all_doc_ids = [row[0] for row in cursor.fetchall()]
                        conn.close()
                    else:
                        # In-memory collection
                        try:
                            all_data = coll.get(include=["ids"])
                            all_doc_ids = all_data.get("ids", [])
                        except Exception:
                            logger.exception("Failed to get IDs from in-memory collection")
                            all_doc_ids = []
                logger.info(f"Retrieved {len(all_doc_ids)} document IDs from database")

            # Map indices to doc_ids and distances to scores
            for i, (qid, dists, idxs) in enumerate(zip(query_ids, distances, indices)):
                query_results = {}
                for dist, idx in zip(dists, idxs):
                    if idx >= 0 and idx < len(all_doc_ids):
                        doc_id = all_doc_ids[idx]
                        # Convert distance to similarity (cosine similarity)
                        # FAISS returns L2 distance for normalized vectors, convert to cosine
                        similarity = 1.0 - (dist / 2.0)  # L2 to cosine for normalized vectors
                        query_results[doc_id] = float(similarity)

                results[qid] = query_results

                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{total} queries")

        except Exception as e:
            logger.error(f"Batch FAISS search failed: {e}")
            # Fallback to sequential retrieval
            logger.warning("Falling back to sequential retrieval...")
            return self.retrieve_bulk(queries, top_k=top_k)

        logger.info(f"Completed optimized retrieval for {total} queries")
        return results

    def retrieve_bulk(
        self, queries: Dict[str, str], top_k: int = 100, batch_size: int = 128
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve for multiple queries.
        Note: DocumentRetriever is optimized for single-query (RAG) flow.
        Bulk retrieval here will be sequential loop over retrieve_top_documents.
        This is slower than raw FAISS batch search but ACCURATE to production.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized.")

        results = {}
        query_ids = list(queries.keys())
        total = len(query_ids)

        logger.info(f"Retrieving for {total} queries using full production pipeline...")

        for i, q_id in enumerate(query_ids):
            query_text = queries[q_id]
            try:
                hits = self.retrieve(query_text, top_k=top_k)
                results[q_id] = {doc_id: float(score) for doc_id, score in hits}
            except Exception as e:
                logger.error(f"Error retrieving for query {q_id}: {e}")
                results[q_id] = {}

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{total} queries")

        return results

    def export_beir_run(self, queries: Dict[str, str], output_file: str, top_k: int = 100):
        """Run retrieval and export in BEIR/TREC format."""
        results = self.retrieve_bulk(queries, top_k=top_k)
        logger.info(f"Saving run to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results
