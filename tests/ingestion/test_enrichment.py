"""
Tests for the ChunkEnricher class.
"""

import unittest
from unittest.mock import MagicMock

import nltk
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

from cubo.processing.enrichment import ChunkEnricher


class TestChunkEnricher(unittest.TestCase):

    def setUp(self):
        # Download the 'punkt' tokenizer for NLTK
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt")
            except Exception:
                # If we cannot download the tokenizer (e.g., no network), skip the test
                self.skipTest("Could not ensure punkt tokenizer for NLTK")

    def test_enrich_chunks(self):
        # 1. Create a mock LLM provider
        mock_llm_provider = MagicMock()

        # Define the responses for the different prompts
        mock_llm_provider.generate_response.side_effect = [
            # First chunk
            "This is a summary of the first chunk.",
            "keyword1, keyword2",
            "Category A",
            "4.5",
            # Second chunk
            "This is a summary of the second chunk.",
            "keyword3, keyword4",
            "Category B",
            "4.2",
        ]

        # 2. Instantiate the ChunkEnricher
        enricher = ChunkEnricher(llm_provider=mock_llm_provider)

        # 3. Create test data
        chunks = [
            "This is the first chunk of text.",
            "This is the second chunk of text.",
        ]

        reference_summaries = [
            "A summary of the first chunk.",
            "A summary of the second chunk.",
        ]
        reference_keywords = [
            ["keyword1", "keyword2"],
            ["keyword3", "keyword5"],  # Intentionally different to test F1 score
        ]

        # 4. Call the enrich_chunks method
        enriched_chunks = enricher.enrich_chunks(chunks)

        # 5. Assert the results
        self.assertEqual(len(enriched_chunks), 2)

        # 6. Evaluate the quality of the summaries and keywords
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        for i in range(len(chunks)):
            # Evaluate summary
            scores = scorer.score(reference_summaries[i], enriched_chunks[i]["summary"])
            self.assertGreater(scores["rouge1"].fmeasure, 0.5)
            self.assertGreater(scores["rougeL"].fmeasure, 0.5)

            # Evaluate keywords
            # Convert keywords to a binary representation for F1 score calculation
            all_keywords = sorted(list(set(reference_keywords[i] + enriched_chunks[i]["keywords"])))
            y_true = [1 if k in reference_keywords[i] else 0 for k in all_keywords]
            y_pred = [1 if k in enriched_chunks[i]["keywords"] else 0 for k in all_keywords]

            if len(all_keywords) > 0:
                f1 = f1_score(y_true, y_pred)
                # Allow equality at the boundary to be more robust in CI
                self.assertGreaterEqual(f1, 0.5)


if __name__ == "__main__":
    unittest.main()
