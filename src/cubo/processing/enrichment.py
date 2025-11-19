"""
This module contains the ChunkEnricher class, which uses an LLM to enrich document chunks
with summaries, keywords, and categories.
"""
from typing import List, Dict

from src.cubo.processing.generator import ResponseGenerator

class ChunkEnricher:
    """
    Enriches document chunks with summaries, keywords, and categories using an LLM.
    """

    def __init__(self, llm_provider: ResponseGenerator):
        self.llm_provider = llm_provider

    def enrich_chunks(self, chunks: List[str]) -> List[Dict]:
        """
        Enriches a list of document chunks.

        :param chunks: A list of text chunks.
        :return: A list of dictionaries, where each dictionary contains the original
                 chunk and the enriched data.
        """
        enriched_chunks = []
        for chunk in chunks:
            summary = self._generate_summary(chunk)
            keywords = self._generate_keywords(chunk)
            category = self._generate_category(chunk)
            consistency_score = self._check_self_consistency(chunk, summary)

            enriched_chunks.append({
                'text': chunk,
                'summary': summary,
                'keywords': keywords,
                'category': category,
                'consistency_score': consistency_score,
            })
        return enriched_chunks

    def _generate_summary(self, chunk: str) -> str:
        """Generates a summary for a single chunk."""
        prompt = f"Summarize the following text in one sentence:\n\n{chunk}"
        return self.llm_provider.generate_response(prompt, "")

    def _generate_keywords(self, chunk: str) -> List[str]:
        """Extracts keywords from a single chunk."""
        prompt = f"Extract the top 5 most important keywords from the following text. Return them as a comma-separated list:\n\n{chunk}"
        keywords_str = self.llm_provider.generate_response(prompt, "")
        return [k.strip() for k in keywords_str.split(',')]

    def _generate_category(self, chunk: str) -> str:
        """Assigns a category to a single chunk."""
        prompt = f"Assign a single category to the following text (e.g., 'Technology', 'Finance', 'Health'):\n\n{chunk}"
        return self.llm_provider.generate_response(prompt, "")

    def _check_self_consistency(self, chunk: str, summary: str) -> float:
        """
        Checks the self-consistency of a summary by asking the LLM to rate it.
        """
        prompt = f"On a scale of 1 to 5, how well does the following summary capture the main points of the original text? Only return the number.\n\nOriginal text: {chunk}\n\nSummary: {summary}"
        try:
            score = float(self.llm_provider.generate_response(prompt, ""))
        except (ValueError, TypeError):
            score = 0.0
        return score
