from __future__ import annotations

import json
from typing import Optional, Tuple

from src.cubo.config import config
from src.cubo.processing.generator import create_response_generator
from src.cubo.utils.logger import logger


class LLMQueryClassifier:
    def __init__(self, model_override: Optional[str] = None, timeout_ms: int = 1000):
        self.model = model_override or config.get('llm_model')
        self.timeout_ms = timeout_ms
        # Use repo LLM generator to keep consistency with generation stack
        self.generator = create_response_generator()

    def _build_prompt(self, query: str) -> str:
        prompt = (
            "You are a classifier. Classify the user's query into exactly one of the following labels: "
            "FACTUAL, CONCEPTUAL, COMPARATIVE, TEMPORAL, EXPLORATORY. "
            "Return valid JSON only, with fields: label (string), confidence (float 0..1), reason (short string).\n"
            f"Query: {query}\n"
            "JSON_OUTPUT:"
        )
        return prompt

    def classify(self, query: str) -> Tuple[str, float, Optional[str]]:
        prompt = self._build_prompt(query)
        try:
            # create_response_generator uses repo LLM provider; we use generate_response to get raw answer
            result = self.generator.generate_response(prompt, context='', messages=None)
            # Try to extract JSON from the response
            text = result.strip()
            # If the model wraps code blocks, try to find JSON inside
            if '```json' in text:
                start = text.index('```json') + len('```json')
                end = text.find('```', start)
                json_text = text[start:end].strip()
            else:
                # often the model outputs JSON directly
                json_text = text
            data = json.loads(json_text)
            label = data.get('label', '').strip().lower()
            conf = float(data.get('confidence', 0.0))
            reason = data.get('reason', '')
            return label, conf, reason
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return 'exploratory', 0.0, None
