import warnings
from typing import List


def test_hybrid_retriever_deprecation_warning():
    # Importing the old module should raise a DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from src.cubo.retrieval.hybrid_retriever import HybridRetriever  # noqa: F401

        assert any(
            issubclass(x.category, DeprecationWarning) for x in w
        ), "No DeprecationWarning raised"


def test_hybrid_retriever_consistency():
    # Create fake BM25 and FAISS components
    class FakeBM25:
        def search(self, query: str, top_k: int = 10):
            return [
                {"doc_id": "a", "similarity": 1.0},
                {"doc_id": "b", "similarity": 0.5},
            ]

    class FakeFaiss:
        def search(self, query_embedding, k=10):
            # return results with id keys
            return [
                {"id": "b", "distance": 0.1},
                {"id": "a", "distance": 0.2},
            ]

    class FakeEmbedGen:
        def encode(self, texts: List[str], batch_size: int = 32):
            return [[0.1, 0.2, 0.3] for _ in texts]

    docs = [
        {"doc_id": "a", "text": "A", "metadata": {}},
        {"doc_id": "b", "text": "B", "metadata": {}},
    ]

    # Import both classes
    from src.cubo.retrieval.hybrid_retriever import HybridRetriever as ShimHybrid
    from src.cubo.retrieval.retriever import FaissHybridRetriever

    # Instantiate both with fake components
    bm25 = FakeBM25()
    faiss = FakeFaiss()
    embed = FakeEmbedGen()

    # Use FaissHybridRetriever
    r1 = FaissHybridRetriever(bm25, faiss, embed, docs)
    # Use shim HybridRetriever (should be same class alias)
    r2 = ShimHybrid(bm25, faiss, embed, docs)

    out1 = r1.search("test", top_k=2)
    out2 = r2.search("test", top_k=2)

    assert len(out1) == len(out2)
    # Compare by doc_id and score
    for a, b in zip(out1, out2):
        assert a["doc_id"] == b["doc_id"]
        assert a.get("score") == b.get("score")
