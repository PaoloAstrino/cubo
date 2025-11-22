"""Local LLM wrapper that exposes the same interface as ResponseGenerator but uses llama_cpp.

This wrapper keeps the API minimal: it exposes `generate_response(query, context, messages=None)` and
implements a synchronous generation via the service_manager to keep parity with the Ollama wrapper.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.cubo.config import config
from src.cubo.services.service_manager import get_service_manager
from src.cubo.utils.logger import logger


class LocalResponseGenerator:
    """Wrapper around llama_cpp's Llama for local model generation."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.get('local_llama_model_path')
        self.service_manager = get_service_manager()
        self._model = None
        if not self.model_path:
            raise RuntimeError("No local Llama model path configured (local_llama_model_path)")
        try:
            # Import lazily to avoid crash if llama_cpp is missing
            from llama_cpp import Llama  # type: ignore
            self._llm = Llama(model_path=self.model_path)
        except Exception as e:
            logger.warning("Failed to initialize local llama model: %s", e)
            self._llm = None

    def generate_response(self, query: str, context: str, messages: List[Dict[str, str]] = None) -> str:
        """Generate a response with local LLM.

        Args:
            query: user query/question
            context: context docs
            messages: optional past conversation messages (not strictly required)
        Returns:
            Generated text response
        """
        convo = []
        if messages is None:
            convo = []
        else:
            convo = messages.copy()

        # Build a simple prompt schema: context + question
        prompt = f"Context: {context}\n\nQuestion: {query}"

        def _generate_operation():
            if not self._llm:
                raise RuntimeError("Local LLM not initialized; check model path and dependencies")
            try:
                # llama_cpp API supports create_completion or generate depending on versions; try both
                if hasattr(self._llm, 'create_completion'):
                    resp = self._llm.create_completion(prompt=prompt, max_tokens=512)
                    # older versions return dict
                    if isinstance(resp, dict):
                        # Support choices structure returned by newer APIs
                        text = resp.get('choices', [{}])[0].get('text') or resp.get('text') or ''
                        return text
                    return str(resp)
                elif hasattr(self._llm, 'generate'):
                    resp = self._llm.generate(prompt)
                    # generate usually returns an object with 'choices'
                    if hasattr(resp, 'choices'):
                        return getattr(resp.choices[0], 'text', '')
                    return str(resp)
                else:
                    # Fallback: try calling model directly
                    return str(self._llm(prompt))
            except Exception as e:
                logger.error("Error during local LLM generation: %s", e)
                raise

        start = time.time()
        result = self.service_manager.execute_sync('llm_generation', _generate_operation)
        logger.info("Local LLM generated response in %.2fs", time.time() - start)
        return result
