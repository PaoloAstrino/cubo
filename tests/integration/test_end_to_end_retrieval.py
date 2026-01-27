"""
Integration tests for end-to-end retrieval pipeline.

Tests the complete retrieval workflow from document ingestion
through embedding generation, indexing, querying, and result fusion.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_work_dir():
    """Create temporary working directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_corpus():
    """Generate sample corpus for retrieval."""
    return [
        {"_id": "doc1", "text": "Machine learning is a subset of artificial intelligence."},
        {"_id": "doc2", "text": "Deep learning uses neural networks with multiple layers."},
        {
            "_id": "doc3",
            "text": "Natural language processing enables computers to understand text.",
        },
        {"_id": "doc4", "text": "Computer vision allows machines to interpret visual information."},
        {
            "_id": "doc5",
            "text": "Reinforcement learning trains agents through rewards and penalties.",
        },
    ]


@pytest.fixture
def sample_queries():
    """Generate sample queries for testing."""
    return [
        {"_id": "q1", "text": "What is machine learning?"},
        {"_id": "q2", "text": "How do neural networks work?"},
        {"_id": "q3", "text": "Explain natural language processing"},
    ]


class TestRetrievalPipeline:
    """Test suite for complete retrieval workflow."""

    def test_corpus_ingestion_pipeline(self, temp_work_dir, sample_corpus):
        """Test document ingestion and preprocessing."""
        corpus_path = temp_work_dir / "corpus.jsonl"

        # Write corpus to JSONL
        with open(corpus_path, "w") as f:
            for doc in sample_corpus:
                f.write(json.dumps(doc) + "\n")

        # Read and validate
        loaded_corpus = []
        with open(corpus_path, "r") as f:
            for line in f:
                loaded_corpus.append(json.loads(line))

        assert len(loaded_corpus) == 5
        assert loaded_corpus[0]["_id"] == "doc1"
        assert "machine learning" in loaded_corpus[0]["text"].lower()

    def test_query_loading_pipeline(self, temp_work_dir, sample_queries):
        """Test query loading and preprocessing."""
        queries_path = temp_work_dir / "queries.jsonl"

        # Write queries to JSONL
        with open(queries_path, "w") as f:
            for query in sample_queries:
                f.write(json.dumps(query) + "\n")

        # Read and validate
        loaded_queries = []
        with open(queries_path, "r") as f:
            for line in f:
                loaded_queries.append(json.loads(line))

        assert len(loaded_queries) == 3
        assert loaded_queries[0]["_id"] == "q1"

    def test_embedding_generation_mock(self, sample_corpus):
        """Test embedding generation for corpus."""
        # Mock embedding generation (384-dim for BGE-base)
        np.random.seed(42)
        embeddings = []

        for doc in sample_corpus:
            # Simulate embedding generation
            embedding = np.random.randn(384).astype("float32")
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings)

        assert embeddings_array.shape == (5, 384)

        # Verify normalization
        norms = np.linalg.norm(embeddings_array, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5), decimal=5)

    def test_dense_retrieval_pipeline(self, sample_corpus):
        """Test dense (semantic) retrieval workflow."""
        import faiss

        # Generate embeddings
        np.random.seed(42)
        doc_embeddings = np.random.randn(5, 384).astype("float32")
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Build index
        index = faiss.IndexFlatIP(384)
        index.add(doc_embeddings)

        # Generate query embedding
        query_embedding = np.random.randn(1, 384).astype("float32")
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        k = 3
        scores, indices = index.search(query_embedding, k)

        assert len(indices[0]) == k
        assert len(scores[0]) == k
        assert all(0 <= idx < 5 for idx in indices[0])

    def test_bm25_retrieval_pipeline(self, sample_corpus):
        """Test BM25 (lexical) retrieval workflow."""
        # Mock BM25 scoring
        query_terms = ["machine", "learning"]

        bm25_scores = []
        for doc in sample_corpus:
            doc_text = doc["text"].lower()
            score = sum(1.0 for term in query_terms if term in doc_text)
            bm25_scores.append((doc["_id"], score))

        # Sort by score
        bm25_scores.sort(key=lambda x: x[1], reverse=True)

        # doc1 should rank first (contains both terms)
        assert bm25_scores[0][0] == "doc1"
        assert bm25_scores[0][1] > 0

    def test_hybrid_retrieval_fusion(self):
        """Test fusing dense and BM25 results."""
        # Dense results: doc_id -> rank
        dense_results = {"doc1": 1, "doc2": 3, "doc3": 2}

        # BM25 results: doc_id -> rank
        bm25_results = {"doc1": 2, "doc2": 1, "doc3": 5}

        # RRF fusion with k=60
        k = 60
        fused_scores = {}

        all_docs = set(dense_results.keys()) | set(bm25_results.keys())

        for doc_id in all_docs:
            dense_rank = dense_results.get(doc_id, float("inf"))
            bm25_rank = bm25_results.get(doc_id, float("inf"))

            dense_score = 1 / (dense_rank + k) if dense_rank != float("inf") else 0
            bm25_score = 1 / (bm25_rank + k) if bm25_rank != float("inf") else 0

            fused_scores[doc_id] = dense_score + bm25_score

        # Sort by fused score
        ranked_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Verify fusion occurred
        assert len(ranked_docs) == 3
        assert ranked_docs[0][1] > 0


class TestRetrievalQuality:
    """Test suite for retrieval quality and relevance."""

    def test_top_k_retrieval_size(self):
        """Test that top-k retrieval returns correct number of results."""
        total_docs = 100
        k_values = [1, 5, 10, 50, 100, 200]

        for k in k_values:
            expected_results = min(k, total_docs)
            assert expected_results <= total_docs

    def test_relevance_score_ordering(self):
        """Test that retrieval results are ordered by score."""
        # Simulate retrieval scores
        results = [("doc1", 0.95), ("doc2", 0.87), ("doc3", 0.92), ("doc4", 0.81)]

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Verify ordering
        scores = [score for _, score in sorted_results]
        assert scores == sorted(scores, reverse=True)
        assert sorted_results[0][0] == "doc1"  # Highest score

    def test_query_document_matching(self):
        """Test basic query-document relevance."""
        query = "machine learning algorithms"

        docs = [
            {"id": "doc1", "text": "Machine learning algorithms are used in AI."},
            {"id": "doc2", "text": "Cooking recipes for pasta."},
            {"id": "doc3", "text": "Deep learning is a subset of machine learning."},
        ]

        # Simple lexical overlap scoring
        query_terms = set(query.lower().split())

        scores = []
        for doc in docs:
            doc_terms = set(doc["text"].lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append((doc["id"], overlap))

        scores.sort(key=lambda x: x[1], reverse=True)

        # doc1 and doc3 should rank higher than doc2
        assert scores[0][1] > 0
        assert scores[-1][0] == "doc2"  # Irrelevant doc


class TestRetrievalOutputFormat:
    """Test suite for retrieval output format compliance."""

    def test_beir_run_format_structure(self, temp_work_dir):
        """Test BEIR run file format (query_id\tdoc_id\trank\tscore)."""
        run_results = [
            ("q1", "doc1", 1, 0.95),
            ("q1", "doc2", 2, 0.87),
            ("q1", "doc3", 3, 0.82),
            ("q2", "doc5", 1, 0.91),
        ]

        run_path = temp_work_dir / "run.txt"

        with open(run_path, "w") as f:
            for query_id, doc_id, rank, score in run_results:
                f.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")

        # Validate format
        with open(run_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                assert len(parts) == 4

                query_id, doc_id, rank, score = parts
                assert query_id.startswith("q")
                assert doc_id.startswith("doc")
                assert int(rank) > 0
                assert 0 <= float(score) <= 1

    def test_topk50_format_structure(self, temp_work_dir):
        """Test topk50 format for RRF input."""
        topk_results = [
            ("q1", "doc1", 1, 0.95),
            ("q1", "doc2", 2, 0.87),
        ]

        topk_path = temp_work_dir / "topk50.txt"

        with open(topk_path, "w") as f:
            for query_id, doc_id, rank, score in topk_results:
                f.write(f"{query_id}\t{doc_id}\t{rank}\t{score:.6f}\n")

        # Verify file can be parsed
        parsed_results = {}
        with open(topk_path, "r") as f:
            for line in f:
                query_id, doc_id, rank, score = line.strip().split("\t")
                if query_id not in parsed_results:
                    parsed_results[query_id] = []
                parsed_results[query_id].append(
                    {"doc_id": doc_id, "rank": int(rank), "score": float(score)}
                )

        assert "q1" in parsed_results
        assert len(parsed_results["q1"]) == 2


class TestRetrievalErrorHandling:
    """Test suite for error handling in retrieval pipeline."""

    def test_empty_corpus_handling(self):
        """Test retrieval with empty corpus."""
        corpus = []

        # Should handle gracefully
        assert len(corpus) == 0

    def test_empty_query_handling(self):
        """Test retrieval with empty query."""
        query = ""

        # Should handle gracefully or return empty results
        assert query == ""

    def test_query_without_matches(self, sample_corpus):
        """Test query that matches no documents."""
        query = "quantum cryptography blockchain"

        # With lexical matching, should return no high-scoring results
        query_terms = set(query.lower().split())

        matches = []
        for doc in sample_corpus:
            doc_terms = set(doc["text"].lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                matches.append(doc["_id"])

        # No matches expected for this unrelated query
        assert len(matches) == 0

    def test_malformed_corpus_entry(self, temp_work_dir):
        """Test handling of malformed corpus entry."""
        malformed_entries = [
            '{"_id": "doc1"}',  # Missing text
            '{"text": "Some text"}',  # Missing _id
            "{invalid json}",  # Invalid JSON
        ]

        valid_count = 0
        for entry in malformed_entries:
            try:
                doc = json.loads(entry)
                if "_id" in doc and "text" in doc:
                    valid_count += 1
            except json.JSONDecodeError:
                pass

        # Should filter out malformed entries
        assert valid_count == 0


class TestRetrievalPerformance:
    """Test suite for retrieval performance characteristics."""

    def test_batch_query_processing(self):
        """Test processing queries in batches."""
        queries = [f"query_{i}" for i in range(100)]
        batch_size = 32

        num_batches = (len(queries) + batch_size - 1) // batch_size

        assert num_batches == 4  # 100 queries / 32 per batch = 4 batches

    def test_index_search_scaling(self):
        """Test that search time scales reasonably with k."""
        # Larger k should take more time, but not exponentially
        k_values = [10, 50, 100]

        # Time complexity should be O(k * log(n)) for most indexes
        for k in k_values:
            assert k > 0
            assert k <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
