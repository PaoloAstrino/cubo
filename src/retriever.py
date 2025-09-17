from typing import List
import time
import torch
import os
import pickle
from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style
from src.config import config
from src.logger import logger
from src.utils import metrics
import chromadb
from chromadb.config import Settings

class DocumentRetriever:
    """Handles document retrieval using semantic similarity with ChromaDB for CUBO."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        settings = Settings(anonymized_telemetry=False)  # Disable telemetry
        self.client = chromadb.PersistentClient(path=config.get("vector_db_path", "./chroma_db"), settings=settings)
        self.collection = self.client.get_or_create_collection("documents")
        self.cache_file = os.path.join(config.get("vector_db_path", "./chroma_db"), "query_cache.pkl")
        self.query_cache = self._load_cache()  # Load persistent cache

    def _load_cache(self):
        """Load cache from disk if exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def add_documents(self, documents: List[str]):
        """Add documents to the vector database in batches."""
        if not documents:
            logger.warning("No documents provided to add")
            return
        try:
            batch_size = config.get("embedding_batch_size", 32)
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                embeddings = self.model.encode(batch).tolist()
                ids = [str(j) for j in range(i, i + len(batch))]
                self.collection.add(embeddings=embeddings, documents=batch, ids=ids)
            logger.info(f"Added {len(documents)} documents to vector DB in batches")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            print(Fore.RED + f"Failed to add documents: {e}" + Style.RESET_ALL)

    def retrieve_top_documents(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve top-k most similar documents from the vector DB with caching."""
        if top_k is None:
            top_k = config.get("top_k", 3)

        if not query.strip():
            logger.warning("Empty query provided")
            print(Fore.YELLOW + "Query is empty. Please provide a valid query." + Style.RESET_ALL)
            return []

        if self.collection.count() == 0:
            logger.warning("No documents in collection")
            print(Fore.YELLOW + "No documents loaded. Please add documents first." + Style.RESET_ALL)
            return []

        # Check cache
        cache_key = (query, top_k)
        if cache_key in self.query_cache:
            logger.info("Retrieved from cache")
            return self.query_cache[cache_key]

        try:
            print(Fore.BLUE + "Retrieving top documents..." + Style.RESET_ALL)
            start = time.time()

            query_embedding = self.model.encode([query]).tolist()[0]
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, include=['distances'])

            # Apply similarity threshold (e.g., cosine distance < 0.5 for relevance)
            threshold = config.get("similarity_threshold", 0.5)
            top_docs = []
            if results['documents'] and results['distances']:
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    if dist < threshold:
                        top_docs.append(doc)

            # Cache the result
            self.query_cache[cache_key] = top_docs
            self._save_cache()  # Persist cache

            print(Fore.GREEN + f"Retrieved in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
            logger.info(f"Retrieved {len(top_docs)} top documents for query (threshold: {threshold})")

            # Record metrics
            duration = time.time() - start
            metrics.record_time("document_retrieval", duration)
            metrics.record_count("retrieval_queries")

            return top_docs

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            print(Fore.RED + f"Retrieval error: {e}" + Style.RESET_ALL)
            return []
