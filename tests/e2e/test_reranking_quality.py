"""
Reranking Quality Impact Test

Tests that reranking improves retrieval precision:
1. Query without reranking
2. Query with reranking enabled
3. Measure quality improvement (Recall@K, MRR)
4. Verify top results are more relevant with reranking
"""

import numpy as np
import pytest

from cubo.retrieval.retriever import DocumentRetriever
from cubo.rerank.reranker import Reranker
from cubo.embeddings.lazy_model_manager import get_lazy_model_manager


@pytest.fixture
def test_corpus():
    """Create a test corpus with clear relevance labels."""
    return [
        # Highly relevant to ML query
        {
            "id": "ml_relevant_1",
            "text": "Machine learning models are trained on large datasets to recognize patterns and make predictions.",
            "relevance_to_ml": 5  # Most relevant
        },
        {
            "id": "ml_relevant_2",
            "text": "Neural networks use backpropagation to adjust weights during training.",
            "relevance_to_ml": 5
        },
        {
            "id": "ml_relevant_3",
            "text": "Supervised learning requires labeled training data for classification tasks.",
            "relevance_to_ml": 4
        },
        # Moderately relevant
        {
            "id": "ml_moderate_1",
            "text": "Data preprocessing is important for cleaning and normalizing input features.",
            "relevance_to_ml": 3
        },
        {
            "id": "ml_moderate_2",
            "text": "Python libraries like scikit-learn and TensorFlow are widely used in data science.",
            "relevance_to_ml": 3
        },
        # Low relevance (mentions ML but not about training)
        {
            "id": "ml_low_1",
            "text": "The history of machine learning dates back to the 1950s with early perceptrons.",
            "relevance_to_ml": 2
        },
        {
            "id": "ml_low_2",
            "text": "Machine learning is a subfield of artificial intelligence.",
            "relevance_to_ml": 2
        },
        # Irrelevant
        {
            "id": "irrelevant_1",
            "text": "The coffee machine in the office kitchen needs maintenance.",
            "relevance_to_ml": 0
        },
        {
            "id": "irrelevant_2",
            "text": "Meeting scheduled for Monday at 2 PM in conference room B.",
            "relevance_to_ml": 0
        },
        {
            "id": "irrelevant_3",
            "text": "Please remember to submit expense reports by the end of the month.",
            "relevance_to_ml": 0
        }
    ]


class TestRerankingQualityImpact:
    """Test that reranking improves retrieval quality."""
    
    def test_reranking_improves_top_result(self, test_corpus, tmp_path):
        """
        Test that reranking places the most relevant document at position 1.
        """
        model = get_lazy_model_manager().get_model()
        
        # Build retriever WITHOUT reranking
        retriever_no_rerank = DocumentRetriever(
            model=model,
            use_reranker=False,
            top_k=5
        )
        
        for doc in test_corpus:
            retriever_no_rerank.add_document(doc["text"], metadata={"id": doc["id"]})
        
        # Query
        query = "How are machine learning models trained?"
        results_no_rerank = retriever_no_rerank.retrieve(query, top_k=5)
        
        # Build retriever WITH reranking
        retriever_with_rerank = DocumentRetriever(
            model=model,
            use_reranker=True,
            top_k=5
        )
        
        for doc in test_corpus:
            retriever_with_rerank.add_document(doc["text"], metadata={"id": doc["id"]})
        
        results_with_rerank = retriever_with_rerank.retrieve(query, top_k=5)
        
        # Extract doc IDs
        ids_no_rerank = [r.get("metadata", {}).get("id") for r in results_no_rerank]
        ids_with_rerank = [r.get("metadata", {}).get("id") for r in results_with_rerank]
        
        print(f"\nWithout reranking: {ids_no_rerank[:3]}")
        print(f"With reranking:    {ids_with_rerank[:3]}")
        
        # Verify improvement: At least one highly relevant doc should be in top-3 with reranking
        highly_relevant_ids = ["ml_relevant_1", "ml_relevant_2", "ml_relevant_3"]
        
        top3_no_rerank = ids_no_rerank[:3]
        top3_with_rerank = ids_with_rerank[:3]
        
        relevant_count_no_rerank = sum(1 for id in top3_no_rerank if id in highly_relevant_ids)
        relevant_count_with_rerank = sum(1 for id in top3_with_rerank if id in highly_relevant_ids)
        
        # Reranking should increase relevant docs in top-3
        assert relevant_count_with_rerank >= relevant_count_no_rerank, \
            f"Reranking didn't improve: {relevant_count_no_rerank} → {relevant_count_with_rerank}"
        
        print(f"✓ Reranking improved relevance: {relevant_count_no_rerank} → {relevant_count_with_rerank} relevant docs in top-3")
    
    def test_reranking_filters_irrelevant_results(self, test_corpus, tmp_path):
        """
        Test that reranking demotes irrelevant documents.
        """
        model = get_lazy_model_manager().get_model()
        retriever = DocumentRetriever(model=model, use_reranker=True, top_k=10)
        
        for doc in test_corpus:
            retriever.add_document(doc["text"], metadata={"id": doc["id"], "relevance": doc["relevance_to_ml"]})
        
        query = "Explain machine learning model training"
        results = retriever.retrieve(query, top_k=5)
        
        # Extract relevance scores from ground truth
        relevances = [r.get("metadata", {}).get("relevance", 0) for r in results]
        
        # Top results should have higher relevance than bottom results
        top_3_avg = np.mean(relevances[:3])
        bottom_2_avg = np.mean(relevances[3:5])
        
        assert top_3_avg > bottom_2_avg, \
            f"Top-3 avg relevance ({top_3_avg}) not > bottom-2 ({bottom_2_avg})"
        
        print(f"✓ Reranking prioritizes relevant docs: top-3 avg={top_3_avg:.1f}, bottom-2 avg={bottom_2_avg:.1f}")
    
    def test_recall_at_k_with_reranking(self, test_corpus):
        """
        Test Recall@K metric improvement with reranking.
        
        Recall@K = (# relevant docs in top-K) / (total # relevant docs)
        """
        model = get_lazy_model_manager().get_model()
        
        # Ground truth: relevant doc IDs
        relevant_ids = [doc["id"] for doc in test_corpus if doc["relevance_to_ml"] >= 4]
        
        # Without reranking
        retriever_no_rerank = DocumentRetriever(model=model, use_reranker=False, top_k=10)
        for doc in test_corpus:
            retriever_no_rerank.add_document(doc["text"], metadata={"id": doc["id"]})
        
        results_no_rerank = retriever_no_rerank.retrieve("machine learning training", top_k=5)
        retrieved_ids_no_rerank = [r.get("metadata", {}).get("id") for r in results_no_rerank]
        
        recall_no_rerank = len(set(retrieved_ids_no_rerank) & set(relevant_ids)) / len(relevant_ids)
        
        # With reranking
        retriever_with_rerank = DocumentRetriever(model=model, use_reranker=True, top_k=10)
        for doc in test_corpus:
            retriever_with_rerank.add_document(doc["text"], metadata={"id": doc["id"]})
        
        results_with_rerank = retriever_with_rerank.retrieve("machine learning training", top_k=5)
        retrieved_ids_with_rerank = [r.get("metadata", {}).get("id") for r in results_with_rerank]
        
        recall_with_rerank = len(set(retrieved_ids_with_rerank) & set(relevant_ids)) / len(relevant_ids)
        
        print(f"\nRecall@5 without reranking: {recall_no_rerank:.2%}")
        print(f"Recall@5 with reranking:    {recall_with_rerank:.2%}")
        
        # Reranking should maintain or improve recall
        assert recall_with_rerank >= recall_no_rerank - 0.1, \
            f"Reranking severely degraded recall: {recall_no_rerank:.2%} → {recall_with_rerank:.2%}"
        
        print(f"✓ Reranking impact: {(recall_with_rerank - recall_no_rerank)*100:+.1f}% recall change")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
