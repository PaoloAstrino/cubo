"""
This module contains the ChunkEnricher class, which uses an LLM to enrich document chunks
with summaries, keywords, and categories. The module avoids importing the full LLM
implementation at runtime when only type checking is used, to keep tests and lightweight
scripts from pulling in heavy dependencies like Ollama.
"""
from typing import List, Dict, TYPE_CHECKING
import re
from src.cubo.utils.logger import logger

if TYPE_CHECKING:
    from src.cubo.processing.generator import ResponseGenerator

class ChunkEnricher:
    """
    Enriches document chunks with summaries, keywords, and categories using an LLM.
    """

    def __init__(self, llm_provider: "ResponseGenerator"):
        self.llm_provider = llm_provider

    def enrich_chunks(self, chunks: List[str]) -> List[Dict]:
        """
        Enriches a list of document chunks.

        :param chunks: A list of text chunks.
        :return: A list of dictionaries, where each dictionary contains the original
                 chunk and the enriched data.
        """
        enriched_chunks = []
        if not chunks:
            return enriched_chunks

        for chunk in chunks:
            try:
                summary = self._generate_summary(chunk)
            except Exception as e:
                logger.warning(f"Failed to generate summary for chunk: {e}")
                summary = ''

            try:
                keywords = self._generate_keywords(chunk)
            except Exception as e:
                logger.warning(f"Failed to extract keywords for chunk: {e}")
                keywords = []

            try:
                category = self._generate_category(chunk)
            except Exception as e:
                logger.warning(f"Failed to generate category for chunk: {e}")
                category = 'general'

            try:
                consistency_score = self._check_self_consistency(chunk, summary)
            except Exception as e:
                logger.warning(f"Failed consistency check for chunk: {e}")
                consistency_score = 0.0

            enriched_chunks.append({
                'text': chunk,
                'summary': summary or '',
                'keywords': keywords or [],
                'category': category or 'general',
                'consistency_score': float(consistency_score) if consistency_score is not None else 0.0,
            })
        return enriched_chunks

    def _generate_summary(self, chunk: str) -> str:
        """Generates a summary for a single chunk.

        Returns an empty string on failure.
        """
        prompt = f"Summarize the following text in one sentence:\n\n{chunk}"
        result = self.llm_provider.generate_response(prompt, "")
        return (result or '').strip()

    def _generate_keywords(self, chunk: str) -> List[str]:
        """Extracts keywords from a single chunk.

        The LLM may return a comma/semicolon/newline separated string. This method
        normalizes the separators and returns a list of unique, stripped keywords.
        """
        prompt = (
            f"Extract the top 5 most important keywords from the following text. "
            f"Return them as a comma-separated list:\n\n{chunk}"
        )
        keywords_str = self.llm_provider.generate_response(prompt, "") or ''

        # Normalize separators (comma, semicolon, pipe, newline)
        parts = re.split(r"[;,\n|]+", keywords_str)
        keywords = [p.strip() for p in parts if p and p.strip()]
        # De-duplicate preserving order
        seen = set()
        deduped = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                deduped.append(kw)
        return deduped

    def _generate_category(self, chunk: str) -> str:
        """Assigns a category to a single chunk.

        Returns a single normalized category name (lowercased) or 'general' on failure.
        """
        prompt = (
            f"Assign a single category to the following text (e.g., 'Technology', 'Finance', 'Health'):\n\n{chunk}"
        )
        result = self.llm_provider.generate_response(prompt, "") or 'general'
        return (result.strip() or 'general')

    def _check_self_consistency(self, chunk: str, summary: str) -> float:
        """
        Checks the self-consistency of a summary by asking the LLM to rate it.
        """
        prompt = (
            f"On a scale of 1 to 5, how well does the following summary capture the main points of the original text? Only return the number.\n\n"
            f"Original text: {chunk}\n\nSummary: {summary}"
        )
        try:
            resp = self.llm_provider.generate_response(prompt, "")
            score = float(resp)
        except (ValueError, TypeError) as e:
            logger.debug(f"Consistency score parsing error: {e}, raw: {resp if 'resp' in locals() else 'N/A'}")
            score = 0.0

        # Clamp the score to [0.0, 5.0]
        try:
            if score < 0.0:
                score = 0.0
            if score > 5.0:
                score = 5.0
        except Exception:
            score = 0.0
        return score
