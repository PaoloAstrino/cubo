import time
from typing import Dict, List, Optional

import ollama
from colorama import Fore, Style

from src.cubo.config import config
from src.cubo.services.service_manager import get_service_manager
from src.cubo.utils.logger import logger
from src.cubo.utils.trace_collector import trace_collector


class ResponseGenerator:
    """Handles response generation using Ollama LLM for CUBO."""

    def __init__(self):
        self.messages = []
        self.service_manager = get_service_manager()
        # Load system prompt from config, with fallback to default
        self.system_prompt = config.get(
            "llm.system_prompt",
            "You are an AI assistant that answers queries strictly based on the "
            "provided context from documents. Do not use any external knowledge, "
            "assumptions, or invented information.",
        )

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
        user_content = f"Context: {context}\n\nQuestion: {query}"
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
            response = ollama.chat(model=model_name, messages=conversation_messages)
            return response["message"]["content"]

        return self.service_manager.execute_sync("llm_generation", _generate_operation)

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
            from src.cubo.processing.llm_local import LocalResponseGenerator

            return LocalResponseGenerator(config.get("local_llama_model_path", None))
        except Exception:
            logger.warning(
                "Failed to initialize LocalResponseGenerator; falling back to default ResponseGenerator"
            )
            return ResponseGenerator()
    else:
        return ResponseGenerator()
