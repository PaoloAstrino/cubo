"""
Vector store abstraction with FAISS as the primary backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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

    def reset(self) -> None:
        raise NotImplementedError()
    def delete(self, ids=None) -> None:
        """Delete ids from the store if supported. Default: NotImplemented."""
        raise NotImplementedError()


class FaissStore(VectorStore):
    def __init__(self, dimension: int, index_dir: Optional[Path] = None, index_root: Optional[Path] = None):
        self.dimension = dimension
        self.index_dir = Path(index_dir) if index_dir else Path(config.get('vector_store_path', './faiss_store'))
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        self._index = FAISSIndexManager(dimension, index_dir=self.index_dir, index_root=index_root)
        self.index_root = index_root
        # local maps: id -> text/metadata
        self._docs: Dict[str, str] = {}
        self._metas: Dict[str, Dict] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._access_counts: Dict[str, int] = {}
        # Configure hot fraction from config
        from src.cubo.config import config as _config
        self.hot_fraction = float(_config.get('vector_index.hot_ratio', 0.2))
        self._index.hot_fraction = self.hot_fraction

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
        # Return structure: 'ids', 'documents', 'metadatas' arrays
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
            # If a 'where' filter is provided, apply it against stored metadata
            if where and isinstance(where, dict):
                for did, doc in self._docs.items():
                    meta = self._metas.get(did, {})
                    match = True
                    for key, val in where.items():
                        # Support simple equality checks
                        if meta.get(key) != val:
                            match = False
                            break
                    if match:
                        ids_out.append(did)
                        docs.append(doc)
                        metas.append(meta)
            else:
                for did, doc in self._docs.items():
                    ids_out.append(did)
                    docs.append(doc)
                    metas.append(self._metas.get(did, {}))
        return {"ids": ids_out, "documents": [docs], "metadatas": [metas]}

    def query(self, query_embeddings=None, n_results=10, include=None, where=None):
        # Query FAISS for nearest neighbors
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

    def reset(self) -> None:
        from src.cubo.indexing.faiss_index import FAISSIndexManager
        self._index = FAISSIndexManager(self.dimension, index_dir=self.index_dir, index_root=getattr(self, 'index_root', None))
        self._index.hot_fraction = self.hot_fraction
        self._docs.clear()
        self._metas.clear()
        self._embeddings.clear()
        self._access_counts.clear()

    def delete(self, ids=None) -> None:
        """Delete entries from the FAISS store by removing them from internal maps and rebuilding the index."""
        if not ids:
            return
        id_set = set(ids)
        # remove from local metadata
        for did in list(self._docs.keys()):
            if did in id_set:
                self._docs.pop(did, None)
                self._metas.pop(did, None)
                self._embeddings.pop(did, None)
                self._access_counts.pop(did, None)
        # Rebuild indexes with remaining data
        remaining_ids = list(self._embeddings.keys())
        vectors = [self._embeddings[did] for did in remaining_ids]
        try:
            self._index.build_indexes(vectors, remaining_ids, append=False)
        except Exception:
            # If rebuild fails, reset the index and fallback to empty
            self.reset()


def create_vector_store(backend: str = None, collection_name: Optional[str] = None, **kwargs) -> VectorStore:
    """Create a FAISS vector store instance.
    
    Args:
        backend: Legacy parameter, ignored (FAISS is always used)
        collection_name: Not used for FAISS
        **kwargs: Additional arguments for FaissStore (dimension, index_dir, index_root)
    
    Returns:
        FaissStore instance
    """
    dimension = kwargs.get('dimension', 1536)
    index_dir_arg = kwargs.get('index_dir', config.get('vector_store_path'))
    index_dir = Path(index_dir_arg) if index_dir_arg else None
    index_root_arg = kwargs.get('index_root', config.get('faiss_index_root', None))
    index_root = Path(index_root_arg) if index_root_arg else None
    return FaissStore(dimension, index_dir=index_dir, index_root=index_root)
