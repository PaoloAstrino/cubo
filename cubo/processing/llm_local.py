"""Local LLM wrapper that exposes the same interface as ResponseGenerator but uses llama_cpp.

This wrapper keeps the API minimal: it exposes `generate_response(query, context, messages=None)` and
implements a synchronous generation via the service_manager to keep parity with the Ollama wrapper.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Iterator, List, Optional

from cubo.config import config
from cubo.config.prompt_defaults import DEFAULT_SYSTEM_PROMPT
from cubo.processing.chat_template_manager import ChatTemplateManager
from cubo.services.service_manager import get_service_manager
from cubo.utils.logger import logger


class LocalResponseGenerator:
    """Wrapper around llama_cpp's Llama for local model generation."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.get("local_llama_model_path")
        self.service_manager = get_service_manager()
        self._model = None
        self.chat_template_manager = ChatTemplateManager()
        if not self.model_path:
            raise RuntimeError("No local Llama model path configured (local_llama_model_path)")
        try:
            # Import lazily to avoid crash if llama_cpp is missing
            from llama_cpp import Llama  # type: ignore

            gpu_layers = config.get("llm.n_gpu_layers", 0)
            self._llm = Llama(model_path=self.model_path, n_ctx=2048, n_gpu_layers=gpu_layers)
        except Exception as e:
            logger.warning("Failed to initialize local llama model: %s", e)
            self._llm = None

    def generate_response(
        self, query: str, context: str, messages: List[Dict[str, str]] = None
    ) -> str:
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
            # Basic system prompt if no history provided
            system_prompt = config.get("llm.system_prompt", DEFAULT_SYSTEM_PROMPT)
            convo.append({"role": "system", "content": system_prompt})
        else:
            convo = messages.copy()
            # ensuring system prompt is there if not present might be good, but
            # relying on what's passed is safer for now.

        # Add current user query with context
        user_content = self.chat_template_manager.format_user_message(context, query)
        convo.append({"role": "user", "content": user_content})

        # Use ChatTemplateManager to format
        # We try to infer model name from path or config to pick the right template
        model_name = self.model_path or config.get("llm_model") or "llama3"
        prompt = self.chat_template_manager.format_chat(convo, model_name=model_name)

        def _generate_operation():
            if not self._llm:
                raise RuntimeError("Local LLM not initialized; check model path and dependencies")
            try:
                # llama_cpp API supports create_completion or generate depending on versions; try both
                if hasattr(self._llm, "create_completion"):
                    resp = self._llm.create_completion(prompt=prompt, max_tokens=512)
                    # older versions return dict
                    if isinstance(resp, dict):
                        # Support choices structure returned by newer APIs
                        text = resp.get("choices", [{}])[0].get("text") or resp.get("text") or ""
                        return text
                    return str(resp)
                elif hasattr(self._llm, "generate"):
                    resp = self._llm.generate(prompt)
                    # generate usually returns an object with 'choices'
                    if hasattr(resp, "choices"):
                        return getattr(resp.choices[0], "text", "")
                    return str(resp)
                else:
                    # Fallback: try calling model directly
                    return str(self._llm(prompt))
            except Exception as e:
                logger.error("Error during local LLM generation: %s", e)
                raise

        start = time.time()
        result = self.service_manager.execute_sync("llm_generation", _generate_operation)
        logger.info("Local LLM generated response in %.2fs", time.time() - start)
        return result

    def generate_response_stream(
        self,
        query: str,
        context: str,
        messages: List[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> Iterator[Dict[str, any]]:
        """Generate a streaming response with local LLM.

        Yields NDJSON events:
        - {'type': 'token', 'delta': '...', 'trace_id': '...'}
        - {'type': 'done', 'answer': '...', 'trace_id': '...', 'duration_ms': 123}
        """
        convo = []
        if messages is None:
            system_prompt = config.get("llm.system_prompt", DEFAULT_SYSTEM_PROMPT)
            convo.append({"role": "system", "content": system_prompt})
        else:
            convo = messages.copy()

        user_content = self.chat_template_manager.format_user_message(context, query)
        convo.append({"role": "user", "content": user_content})

        model_name = self.model_path or config.get("llm_model") or "llama3"
        prompt = self.chat_template_manager.format_chat(convo, model_name=model_name)

        start_time = time.time()

        if not self._llm:
            logger.error("Local LLM not initialized")
            yield {
                "type": "error",
                "message": "Local LLM not initialized; check model path and dependencies",
                "trace_id": trace_id,
            }
            return

        try:
            # Try streaming via create_completion with stream=True
            if hasattr(self._llm, "create_completion"):
                accumulated = []
                try:
                    stream = self._llm.create_completion(prompt=prompt, max_tokens=512, stream=True)
                    for chunk in stream:
                        if isinstance(chunk, dict):
                            delta = chunk.get("choices", [{}])[0].get("text", "")
                            if delta:
                                accumulated.append(delta)
                                yield {"type": "token", "delta": delta, "trace_id": trace_id}

                    answer = "".join(accumulated)
                    duration_ms = int((time.time() - start_time) * 1000)
                    yield {
                        "type": "done",
                        "answer": answer,
                        "trace_id": trace_id,
                        "duration_ms": duration_ms,
                    }
                    logger.info("Local LLM streamed response in %.2fs", time.time() - start_time)
                    return
                except Exception as stream_err:
                    logger.warning(f"Streaming failed, falling back: {stream_err}")

            # Fallback: generate synchronously and chunk the result
            answer = self.generate_response(query, context, messages)
            duration_ms = int((time.time() - start_time) * 1000)

            # Chunk answer into tokens for streaming UX
            chunk_size = 10
            for i in range(0, len(answer), chunk_size):
                delta = answer[i : i + chunk_size]
                yield {"type": "token", "delta": delta, "trace_id": trace_id}

            yield {
                "type": "done",
                "answer": answer,
                "trace_id": trace_id,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Local LLM streaming error: {e}")
            yield {"type": "error", "message": str(e), "trace_id": trace_id}
