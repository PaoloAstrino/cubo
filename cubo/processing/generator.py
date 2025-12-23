import time
from typing import Dict, Iterator, List, Optional

try:
    import ollama

    OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    OLLAMA_AVAILABLE = False
    # Import failure is OK; tests can run without Ollama installed. We'll fallback or stub when needed.
from colorama import Fore, Style

from cubo.config import config
from cubo.config.prompt_defaults import DEFAULT_SYSTEM_PROMPT
from cubo.processing.chat_template_manager import ChatTemplateManager
from cubo.services.service_manager import get_service_manager
from cubo.utils.logger import logger
from cubo.utils.trace_collector import trace_collector


class ResponseGenerator:
    """Handles response generation using Ollama LLM for CUBO."""

    def __init__(self):
        self.messages = []
        self.service_manager = get_service_manager()
        self.chat_template_manager = ChatTemplateManager()
        # Load system prompt from config, with fallback to canonical default
        self.system_prompt = config.get("llm.system_prompt", DEFAULT_SYSTEM_PROMPT)

    def initialize_conversation(self):
        """Initialize the conversation with system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_response(
        self,
        query: str,
        context: str,
        messages: List[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Generate a response using the LLM."""
        conversation_messages = self._prepare_conversation_messages(messages)
        self._add_user_message(conversation_messages, query, context)

        start_time = time.time()
        print(Fore.BLUE + "Generating response..." + Style.RESET_ALL)

        assistant_content = self._generate_with_ollama(conversation_messages)

        self._update_conversation_history(conversation_messages, assistant_content, messages)
        duration_ms = int((time.time() - start_time) * 1000)
        self._log_generation_time(start_time)
        if trace_id:
            try:
                trace_collector.record(
                    trace_id, "generator", "generator.generated", {"duration_ms": duration_ms}
                )
            except Exception:
                pass

        return assistant_content

    def generate_text(self, prompt: str) -> str:
        """Generate free-form text from a single prompt without chat templating."""

        conversation_messages = [{"role": "user", "content": prompt}]
        return self._generate_with_ollama(conversation_messages)

    def _prepare_conversation_messages(
        self, messages: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Prepare the conversation messages for generation."""
        if messages is None:
            if not self.messages:
                self.initialize_conversation()
            return self.messages.copy()  # Don't modify the original
        else:
            return messages.copy() if messages else []

    def _add_user_message(
        self, conversation_messages: List[Dict[str, str]], query: str, context: str
    ):
        """Add the user message with context to the conversation."""
        user_content = self.chat_template_manager.format_user_message(context, query)
        conversation_messages.append({"role": "user", "content": user_content})

    def _generate_with_ollama(self, conversation_messages: List[Dict[str, str]]) -> str:
        """Generate response using Ollama API."""

        def _generate_operation():
            # Try multiple config keys for model name
            model_name = (
                config.get("llm.model_name")
                or config.get("selected_llm_model")
                or config.get("llm_model")
                or "llama3"  # Default fallback
            )
            if not OLLAMA_AVAILABLE or ollama is None:
                raise RuntimeError(
                    "Ollama Python package not available. Install ollama or configure local provider."
                )
            response = ollama.chat(model=model_name, messages=conversation_messages)
            return response["message"]["content"]

        return self.service_manager.execute_sync("llm_generation", _generate_operation)

    def generate_response_stream(
        self,
        query: str,
        context: str,
        messages: List[Dict[str, str]] = None,
        trace_id: Optional[str] = None,
    ) -> Iterator[Dict[str, any]]:
        """Generate a streaming response using the LLM.

        Yields NDJSON events:
        - {'type': 'token', 'delta': '...', 'trace_id': '...'}
        - {'type': 'done', 'answer': '...', 'trace_id': '...', 'duration_ms': 123}
        """
        conversation_messages = self._prepare_conversation_messages(messages)
        self._add_user_message(conversation_messages, query, context)

        start_time = time.time()
        print(Fore.BLUE + "Generating streaming response..." + Style.RESET_ALL)

        model_name = (
            config.get("llm.model_name")
            or config.get("selected_llm_model")
            or config.get("llm_model")
            or "llama3"
        )

        if not OLLAMA_AVAILABLE or ollama is None:
            # Fallback: yield single final event
            logger.warning("Ollama not available; falling back to non-streaming")
            answer = self._generate_with_ollama(conversation_messages)
            duration_ms = int((time.time() - start_time) * 1000)
            yield {
                "type": "done",
                "answer": answer,
                "trace_id": trace_id,
                "duration_ms": duration_ms,
            }
            return

        # Use Ollama streaming
        try:
            accumulated = []
            for chunk in ollama.chat(model=model_name, messages=conversation_messages, stream=True):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    accumulated.append(delta)
                    yield {"type": "token", "delta": delta, "trace_id": trace_id}

            assistant_content = "".join(accumulated)
            self._update_conversation_history(conversation_messages, assistant_content, messages)
            duration_ms = int((time.time() - start_time) * 1000)

            if trace_id:
                try:
                    trace_collector.record(
                        trace_id,
                        "generator",
                        "generator.stream_completed",
                        {"duration_ms": duration_ms, "tokens": len(accumulated)},
                    )
                except Exception:
                    pass

            yield {
                "type": "done",
                "answer": assistant_content,
                "trace_id": trace_id,
                "duration_ms": duration_ms,
            }
            self._log_generation_time(start_time)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"type": "error", "message": str(e), "trace_id": trace_id}

    def _update_conversation_history(
        self,
        conversation_messages: List[Dict[str, str]],
        assistant_content: str,
        messages: List[Dict[str, str]] = None,
    ):
        """Update the conversation history if using instance messages."""
        if messages is None:
            conversation_messages.append({"role": "assistant", "content": assistant_content})
            self.messages = conversation_messages

    def _log_generation_time(self, start_time: float):
        """Log the generation time and success message."""
        duration = time.time() - start_time
        print(Fore.GREEN + f"Response generated in {duration:.2f} seconds." + Style.RESET_ALL)
        logger.info("Response generated successfully")


def create_response_generator() -> ResponseGenerator:
    """Factory that selects an LLM provider based on configuration.

    Returns:
        An instance implementing the same interface as ResponseGenerator
    """
    provider = config.get("llm.provider", "ollama")
    if provider == "local":
        try:
            from cubo.processing.llm_local import LocalResponseGenerator

            return LocalResponseGenerator(config.get("local_llama_model_path", None))
        except Exception:
            logger.warning(
                "Failed to initialize LocalResponseGenerator; falling back to default ResponseGenerator"
            )
            return ResponseGenerator()
    else:
        # If provider is 'ollama' but the package is not available, attempt to fallback to local provider
        if provider in ("ollama", "default") and not OLLAMA_AVAILABLE:
            logger.warning(
                "Ollama provider configured but 'ollama' package not available; defaulting to local provider."
            )
            try:
                from cubo.processing.llm_local import LocalResponseGenerator

                return LocalResponseGenerator(config.get("local_llama_model_path", None))
            except Exception:
                logger.warning(
                    "Local LLama generator failed to initialize; returning default ResponseGenerator which will raise when used."
                )
        return ResponseGenerator()
