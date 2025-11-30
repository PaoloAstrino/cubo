from pathlib import Path

from cubo.indexing.faiss_index import FAISSIndexManager
from cubo.retrieval.bm25_searcher import BM25Searcher
from cubo.retrieval.retriever import FaissHybridRetriever


class FakeEmbeddingGenerator:
    def __init__(self, dim=2):
        self.dim = dim

    def encode(self, texts, batch_size=32):
        return [[float(i % 10) for _ in range(self.dim)] for i, _ in enumerate(texts)]


def _small_docs():
    docs = []
    for i in range(6):
        docs.append(
            {
                "doc_id": f"d{i}",
                "text": f"This is document {i} about apples and bananas",
                "metadata": {},
            }
        )
    return docs


def test_hybrid_with_strategy(tmp_path: Path):
    # Setup fake BM25 and FAISS (use small in-memory fallback)
    bm25 = BM25Searcher(bm25_stats=None)
    # Create an FAISS manager with small vector support
    faiss_manager = FAISSIndexManager(dimension=2, index_dir=tmp_path)
    emb = FakeEmbeddingGenerator(dim=2)
    docs = _small_docs()
    # Build an index with embeddings
    vectors = [[float(i), float(i)] for i in range(len(docs))]
    ids = [d["doc_id"] for d in docs]
    faiss_manager.build_indexes(vectors, ids)
    faiss_manager.save(path=tmp_path)

    class FakeReranker:
        def __init__(self):
            self.called = False

        def rerank(self, query, candidates, max_results=None):
            # Simple rerank: reverse candidate order and attach score
            self.called = True
            out = []
            for i, c in enumerate(reversed(candidates)):
                c2 = c.copy()
                # provide a heuristic rerank score
                c2["rerank_score"] = float(len(candidates) - i) / len(candidates)
                out.append(c2)
            return out

    reranker = FakeReranker()
    hybrid = FaissHybridRetriever(bm25, faiss_manager, emb, documents=docs, reranker=reranker)
    strategy = {"bm25_weight": 0.9, "dense_weight": 0.1, "k_candidates": 5, "use_reranker": False}
    results = hybrid.search("What is apple", top_k=3, strategy=strategy)
    assert isinstance(results, list)
    assert len(results) <= 3

    # reranker not used yet
    assert not reranker.called

    # Now use reranker and assert it's invoked and affects order
    strategy["use_reranker"] = True
    reranked_results = hybrid.search("What is apple", top_k=3, strategy=strategy)
    assert reranker.called
    # Because we reversed candidates in FakeReranker, reranked order should differ or have rerank_score
    assert any("rerank_score" in r for r in reranked_results)
