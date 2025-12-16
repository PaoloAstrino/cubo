import os
import json
import logging
import sqlite3
import shutil
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import numpy as np

from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.retrieval.retriever import DocumentRetriever
from cubo.config import config
from cubo.config.settings import settings

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
        lightweight: bool = False
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
            self.embedding_generator._load_model()
            
        # 2. Initialize Retriever
        # We use the settings from config/settings.py to match production
        self.retriever = DocumentRetriever(
            model=self.embedding_generator.model, # Pass the underlying SentenceTransformer
            top_k=settings.retrieval.default_top_k,
            window_size=settings.retrieval.default_window_size,
            use_sentence_window=config.get("chunking.use_sentence_window", True),
            use_reranker=config.get("retrieval.use_reranker", True),
            # Disable auto-merging for BEIR as it expects flat passages usually
            use_auto_merging=False 
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
            if hasattr(self.retriever, "collection") and hasattr(self.retriever.collection, "load"):
                 self.retriever.collection.load(self.index_dir)

    def index_corpus(
        self, 
        corpus_path: str, 
        index_dir: str, 
        batch_size: int = 512, 
        limit: Optional[int] = None,
        normalize: bool = True
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
            dimension=self.embedding_generator.dimension,
            index_type="hnsw", 
            metric="cosine"
        )
        
        db_path = index_path / "documents.db"
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT, metadata TEXT)")
        # Also create collections table as DocumentStore might check it
        c.execute("CREATE TABLE IF NOT EXISTS collections (id TEXT PRIMARY KEY, name TEXT, created_at TEXT, color TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, vector BLOB, dtype TEXT, dim INTEGER, created_at TEXT)")
        
        count = 0
        batch_docs = []
        batch_ids = []
        
        logger.info(f"Indexing corpus from {corpus_path}...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                try:
                    doc = json.loads(line)
                    doc_id = f"beir_{doc.get('_id')}"
                    text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    if not text: continue
                    batch_docs.append(text)
                    batch_ids.append(doc_id)
                    if len(batch_docs) >= batch_size:
                        self._process_batch(batch_docs, batch_ids, conn, normalize, faiss_manager)
                        count += len(batch_docs)
                        batch_docs = []
                        batch_ids = []
                        if count % 10000 == 0: logger.info(f"Indexed {count} documents...")
                except json.JSONDecodeError: continue
                    
        if batch_docs:
            self._process_batch(batch_docs, batch_ids, conn, normalize, faiss_manager)
            count += len(batch_docs)
            
        logger.info("Saving FAISS index...")
        faiss_manager.save(str(index_path))
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
        embeddings = self.embedding_generator.encode(texts)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
        faiss_manager.add_vectors(ids, embeddings)
        c = conn.cursor()
        data = []
        for doc_id, text in zip(ids, texts):
            # We store metadata as JSON string
            data.append((doc_id, text, json.dumps({"id": doc_id, "source": "beir"})))
        c.executemany("INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)", data)
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
            raise ValueError("Retriever not initialized. Call load_index() or init with lightweight=False.")
            
        # Use the full production pipeline
        # retrieve_top_documents returns List[Dict] with 'id', 'similarity', 'document', etc.
        results = self.retriever.retrieve_top_documents(query, top_k=top_k)
        
        # Convert to (doc_id, score) tuples for BEIR
        return [(r['id'], r['similarity']) for r in results]

    def retrieve_bulk(self, queries: Dict[str, str], top_k: int = 100, batch_size: int = 128) -> Dict[str, Dict[str, float]]:
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
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        return results
