"""
Whoosh-backed BM25 store implementation.
"""
from typing import List, Dict, Optional
from src.cubo.retrieval.bm25_store import BM25Store

try:
    from whoosh import index as whoosh_index
    from whoosh.fields import Schema, ID, TEXT
    from whoosh.qparser import MultifieldParser
    from whoosh.scoring import BM25F
    WHOOSH_AVAILABLE = True
except Exception:
    WHOOSH_AVAILABLE = False


class BM25WhooshStore(BM25Store):
    def __init__(self, index_dir: str = './whoosh_index'):
        if not WHOOSH_AVAILABLE:
            raise ImportError('Whoosh is not installed; install `whoosh` to use this backend')
        self.index_dir = index_dir
        self._ix = None
        self._schema = Schema(doc_id=ID(stored=True, unique=True), text=TEXT(stored=True))
        self._init_index()

    def _init_index(self):
        import os
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir, exist_ok=True)
            self._ix = whoosh_index.create_in(self.index_dir, schema=self._schema)
        else:
            if whoosh_index.exists_in(self.index_dir):
                self._ix = whoosh_index.open_dir(self.index_dir)
            else:
                self._ix = whoosh_index.create_in(self.index_dir, schema=self._schema)

    def index_documents(self, docs: List[Dict]):
        writer = self._ix.writer()
        # Replace index: remove all existing and add
        for doc in docs:
            writer.update_document(doc_id=doc.get('doc_id'), text=doc.get('text', ''))
        writer.commit()
        self.docs = docs

    def add_documents(self, docs: List[Dict], reset: bool = False):
        writer = self._ix.writer()
        for doc in docs:
            writer.update_document(doc_id=doc.get('doc_id'), text=doc.get('text', ''))
        writer.commit()
        # naive append
        if reset:
            self.docs = docs
        else:
            self.docs = self.docs + docs if hasattr(self, 'docs') else docs

    def search(self, query: str, top_k: int = 10, docs: Optional[List[Dict]] = None) -> List[Dict]:
        qp = MultifieldParser(['text'], schema=self._schema)
        q = qp.parse(query)
        results = []
        with self._ix.searcher(weighting=BM25F()) as searcher:
            hits = searcher.search(q, limit=top_k)
            for hit in hits:
                results.append({'doc_id': hit['doc_id'], 'similarity': float(hit.score), 'metadata': {}, 'text': hit['text']})
        return results

    def compute_score(self, query_terms: List[str], doc_id: str, doc_text: Optional[str] = None) -> float:
        # Whoosh scoring is handled at search time; we can provide a heuristic using search of doc_id
        q = ' '.join(query_terms)
        results = self.search(q, top_k=1)
        if results and results[0]['doc_id'] == doc_id:
            return results[0]['similarity']
        return 0.0

    def load_stats(self, path: str):
        # Whoosh stores index files; no JSON stats to load
        return

    def save_stats(self, path: str):
        # Not required for Whoosh
        return

    def close(self):
        # nothing to close
        return
