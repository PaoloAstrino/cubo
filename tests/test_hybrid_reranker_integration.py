from pathlib import Path
from src.cubo.retrieval.retriever import FaissHybridRetriever
from src.cubo.retrieval.bm25_searcher import BM25Searcher
from src.cubo.indexing.faiss_index import FAISSIndexManager
from src.cubo.embeddings.embedding_generator import EmbeddingGenerator


class FakeEmbeddingGenerator:
    def __init__(self, dim=2):
        self.dim = dim

    def encode(self, texts, batch_size=32):
        # simple embedding: map each text to a vector based on its index
        return [[float(i % 10) for _ in range(self.dim)] for i, _ in enumerate(texts)]


def _small_docs():
    docs = []
    docs.append({'doc_id': 'd0', 'text': 'Document about apples and fruit', 'metadata': {}})
    docs.append({'doc_id': 'd1', 'text': 'Document about cars and vehicles', 'metadata': {}})
    docs.append({'doc_id': 'd2', 'text': 'Document about bananas and fruit', 'metadata': {}})
    return docs


def test_reranker_integration_changes_order(tmp_path: Path):
    # Setup fake BM25 and FAISS (use small in-memory fallback)
    bm25 = BM25Searcher(bm25_stats=None)
    # Create an FAISS manager with small vector support
    faiss_manager = FAISSIndexManager(dimension=2, index_dir=tmp_path)
    emb = FakeEmbeddingGenerator(dim=2)
    docs = _small_docs()
    # Build an index with embeddings so FAISS returns consistent ordering
    vectors = [[0.9, 0.9], [0.1, 0.1], [0.8, 0.8]]
    ids = [d['doc_id'] for d in docs]
    faiss_manager.build_indexes(vectors, ids)
    faiss_manager.save(path=tmp_path)

    class FakeCrossEncoderReranker:
        def __init__(self):
            self.called = False

        def rerank(self, query, candidates, max_results=None):
            # This fake reranker promotes 'd1' to top by artificially boosting it
            self.called = True
            out = []
            for i, c in enumerate(candidates):
                # prefer doc_id so mapping is consistent
                doc_id = c.get('doc_id') or c.get('metadata', {}).get('chunk_id') or c.get('document')
                new = c.copy()
                if doc_id == 'd1':
                    new['rerank_score'] = 10.0
                else:
                    new['rerank_score'] = 1.0
                out.append(new)
            # sort by rerank_score desc
            out.sort(key=lambda x: x['rerank_score'], reverse=True)
            return out

    reranker = FakeCrossEncoderReranker()
    hybrid = FaissHybridRetriever(bm25, faiss_manager, emb, documents=docs, reranker=reranker)

    # No reranking: default strategy has use_reranker False
    strategy = {'bm25_weight': 0.5, 'dense_weight': 0.5, 'k_candidates': 5, 'use_reranker': False}
    results = hybrid.search('apple', top_k=3, strategy=strategy)
    assert isinstance(results, list)
    assert len(results) <= 3

    # Ensure reranker not called when disabled
    assert reranker.called is False

    # Now enable reranking
    strategy['use_reranker'] = True
    reranked_results = hybrid.search('apple', top_k=3, strategy=strategy)
    # reranker was called
    assert reranker.called is True
    # Because FakeCrossEncoderReranker promotes 'd1', check it is in the top result
    assert len(reranked_results) > 0
    assert reranked_results[0]['doc_id'] == 'd1' or reranked_results[0].get('metadata', {}).get('doc_id') == 'd1'
