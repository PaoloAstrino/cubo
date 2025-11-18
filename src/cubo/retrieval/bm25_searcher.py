"""
Simple BM25 searcher for fast-pass JSONL output and stats.
"""
from pathlib import Path
import json
import math
from collections import defaultdict, Counter
from typing import List, Dict

from src.cubo.utils.logger import logger


class BM25Searcher:
    def __init__(self, chunks_jsonl: str, bm25_stats: str):
        self.chunks_jsonl = Path(chunks_jsonl)
        self.bm25_stats = Path(bm25_stats)
        self.docs = []  # List[dict] with 'doc_id', 'text', 'metadata'
        self.doc_term_freq = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0.0
        self.term_doc_freq = defaultdict(int)
        self._load()

    def _load(self):
        # Load chunks JSONL
        if not self.chunks_jsonl.exists():
            raise FileNotFoundError(f"Chunks JSONL not found: {self.chunks_jsonl}")

        with open(self.chunks_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                # Create doc_id from file_hash + chunk_index or filename
                file_hash = rec.get('file_hash', '')
                chunk_index = rec.get('chunk_index', 0)
                filename = rec.get('filename', 'unknown')
                doc_id = (file_hash + f"_{chunk_index}") if file_hash else f"{filename}_{chunk_index}"
                text = rec.get('text', rec.get('document', ''))
                self.docs.append({'doc_id': doc_id, 'text': text, 'metadata': rec})

        # Load BM25 stats if present
        if self.bm25_stats.exists():
            with open(self.bm25_stats, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.doc_lengths = {k: int(v) for k, v in data.get('doc_lengths', {}).items()}
                self.avg_doc_length = float(data.get('avg_doc_length', 0.0))
                self.term_doc_freq = defaultdict(int, {k: int(v) for k, v in data.get('term_doc_freq', {}).items()})
                self.doc_term_freq = {k: {tk: int(tv) for tk, tv in v.items()} for k, v in data.get('doc_term_freq', {}).items()}
        else:
            # Stateless fallback: build on the fly
            for d in self.docs:
                tokens = self._tokenize(d['text'])
                tf = Counter(tokens)
                self.doc_term_freq[d['doc_id']] = dict(tf)
                self.doc_lengths[d['doc_id']] = len(tokens)
                for term in set(tokens):
                    self.term_doc_freq[term] += 1
            if self.doc_lengths:
                self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def _tokenize(self, text: str):
        tokens = [w for w in ''.join(c if c.isalnum() else ' ' for c in text.lower()).split() if len(w) > 2]
        return tokens

    def _compute_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        doc_len = self.doc_lengths.get(doc_id, 0)
        if doc_len == 0:
            return 0.0

        total_docs = len(self.doc_lengths) if self.doc_lengths else 1
        avg_len = self.avg_doc_length if self.avg_doc_length > 0 else doc_len

        k1 = 1.5
        b = 0.75
        score = 0.0
        doc_terms = self.doc_term_freq.get(doc_id, {})
        if not doc_terms:
            # Fallback: build term frequencies from text if not precomputed
            text = next((d['text'] for d in self.docs if d['doc_id'] == doc_id), '')
            tokens = self._tokenize(text)
            doc_terms = dict(Counter(tokens))

        for term in query_terms:
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue
            df = self.term_doc_freq.get(term, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_len))
            term_score = idf * (numerator / denominator)
            score += term_score
        return score

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        results = []
        for d in self.docs:
            doc_id = d['doc_id']
            bm25_score = self._compute_bm25_score(query_terms, doc_id)
            norm_score = min(bm25_score / 15.0, 1.0)
            if norm_score > 0.0:
                results.append({'doc_id': doc_id, 'similarity': norm_score, 'metadata': d['metadata'], 'text': d['text']})
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
