"""
Vector store abstraction to allow swapping FAISS and ChromaDB backends.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pathlib import Path
import os

from src.cubo.utils.logger import logger
from src.cubo.config import config


class VectorStore:
    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()

    def get(self, include=None, where=None, ids=None):
        raise NotImplementedError()

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        raise NotImplementedError()

    def save(self, path: Optional[Path] = None) -> None:
        pass

    def load(self, path: Optional[Path] = None) -> None:
        pass


class FaissStore(VectorStore):
    def __init__(self, dimension: int, index_dir: Optional[Path] = None):
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        self._index = FAISSIndexManager(dimension, index_dir=index_dir)
        # local maps: id -> text/metadata
        self._docs: Dict[str, str] = {}
        self._metas: Dict[str, Dict] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._access_counts: Dict[str, int] = {}
        # Configure hot fraction from config
        from src.cubo.config import config as _config
        hot_fraction = float(_config.get('vector_index.hot_ratio', 0.2))
        self._index.hot_fraction = hot_fraction

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        # Persist to FAISS and store metadata locally
        if not embeddings or not ids:
            return
        self._index.build_indexes(embeddings, ids, append=True)
        # Store embeddings for potential rebuilds (e.g., promotion to hot)
        for i, did in enumerate(ids):
            self._embeddings[did] = embeddings[i]
            self._access_counts.setdefault(did, 0)
        for i, did in enumerate(ids):
            self._docs[did] = documents[i] if documents and i < len(documents) else ''
            self._metas[did] = metadatas[i] if metadatas and i < len(metadatas) else {}

    def count(self) -> int:
        return len(self._docs)

    def get(self, include=None, where=None, ids=None):
        # Behavior: return 'ids', 'documents', 'metadatas' arrays; include repeated as lists within lists similar to chroma
        ids_out = []
        docs = []
        metas = []
        if ids:
            for did in ids:
                if did in self._docs:
                    ids_out.append(did)
                    docs.append(self._docs.get(did, ''))
                    metas.append(self._metas.get(did, {}))
        else:
            for did, doc in self._docs.items():
                ids_out.append(did)
                docs.append(doc)
                metas.append(self._metas.get(did, {}))
        return {"ids": ids_out, "documents": [docs], "metadatas": [metas]}

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        # Query FAISS for nearest neighbors; return structure matching chroma API
        if not query_embeddings or self.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        results = self._index.search(query_embeddings[0], k=n_results)
        # map results to documents/metas/ids/distance
        docs = []
        metas = []
        dists = []
        ids_list = []
        for res in results:
            did = res['id']
            docs.append(self._docs.get(did, ''))
            metas.append(self._metas.get(did, {}))
            dists.append(res['distance'])
            ids_list.append(did)
            # track access counts to potentially promote to hot
            self._access_counts[did] = self._access_counts.get(did, 0) + 1
            from src.cubo.config import config as _config
            threshold = int(_config.get('vector_index.promote_threshold', 10))
            if self._access_counts[did] >= threshold:
                try:
                    self.promote_to_hot(did)
                except Exception:
                    # Don't fail queries due to promotion operations
                    pass
        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids_list]}

    def promote_to_hot(self, doc_id: str) -> None:
        """Promote a cold doc into the hot set by rebuilding indexes with the doc first.

        Note: This is a simplistic implementation that rebuilds FAISS indexes.
        """
        if doc_id not in self._embeddings:
            return
        # Rebuild with doc_id among the first hot elements
        # Keep order: put current hot_ids first (if known), then this doc id
        all_ids = list(self._embeddings.keys())
        # Put promoted doc at the front to ensure it's in the hot set after build
        if doc_id in all_ids:
            all_ids.remove(doc_id)
        new_ids = [doc_id] + all_ids
        # Build corresponding embeddings array
        vectors = [self._embeddings[did] for did in new_ids]
        self._index.build_indexes(vectors, new_ids, append=False)
        # Reset access count so promotions don't immediately re-trigger
        self._access_counts[doc_id] = 0

    def save(self, path: Optional[Path] = None) -> None:
        self._index.save(path)

    def load(self, path: Optional[Path] = None) -> None:
        self._index.load(path)


class ChromaStore(VectorStore):
    def __init__(self, db_path: str = None):
        import chromadb
        from chromadb.config import Settings
        p = db_path or config.get('chroma_db_path', './chroma_db')
        self.client = chromadb.PersistentClient(path=p)
        # create/get collection
        name = config.get('collection_name', 'cubo_documents')
        try:
            self.collection = self.client.get_or_create_collection(name=name)
        except Exception:
            # fallback to in-memory collection
            self.collection = None

    def add(self, embeddings=None, documents=None, metadatas=None, ids=None):
        if self.collection:
            self.collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)

    def count(self) -> int:
        if self.collection:
            return self.collection.count()
        return 0

    def get(self, include=None, where=None, ids=None):
        if self.collection:
            return self.collection.get(include=include, where=where, ids=ids)
        return {"ids": [], "documents": [[]], "metadatas": [[]]}

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        if self.collection:
            return self.collection.query(query_embeddings=query_embeddings, n_results=n_results, include=include, where=where)
        return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}


def create_vector_store(backend: str = None, **kwargs) -> VectorStore:
    backend = backend or config.get('vector_store_backend', 'faiss')
    if backend == 'faiss':
        dimension = kwargs.get('dimension', 1536)
        index_dir = Path(kwargs.get('index_dir', config.get('vector_store_path')))
        return FaissStore(dimension, index_dir=index_dir)
    elif backend == 'chroma':
        db_path = kwargs.get('db_path', config.get('chroma_db_path', './chroma_db'))
        return ChromaStore(db_path)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
