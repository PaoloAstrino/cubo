import ollama
import time
from typing import List, Dict
from colorama import Fore, Style
from src.config import config
from src.logger import logger
from src.service_manager import get_service_manager


class ResponseGenerator:
    """Handles response generation using Ollama LLM for CUBO."""

    def __init__(self):
        self.messages = []
        self.service_manager = get_service_manager()
        self.system_prompt = ("You are an AI assistant that answers queries strictly based on the "
                              "provided context from documents. Do not use any external knowledge, "
                              "assumptions, or invented information. If the context does not contain "
                              "relevant information to answer the query, respond with: 'The provided "
                              "documents do not contain information to answer this query.'")

    def initialize_conversation(self):
        """Initialize the conversation with system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_response(self, query: str, context: str, messages: List[Dict[str, str]] = None) -> str:
        """Generate a response using the LLM."""
        def _generate_operation():
            # Use provided messages or initialize new conversation
            if messages is None:
                if not self.messages:
                    self.initialize_conversation()
                conversation_messages = self.messages.copy()  # Don't modify the original
            else:
                conversation_messages = messages.copy() if messages else []

            print(Fore.BLUE + "Generating response..." + Style.RESET_ALL)
            start = time.time()

            # Add user message with context
            user_content = f"Context: {context}\n\nQuestion: {query}"
            conversation_messages.append({"role": "user", "content": user_content})

            # Generate response using Ollama
            model_name = config.get("selected_llm_model") or config.get("llm_model")
            response = ollama.chat(model=model_name, messages=conversation_messages)
            assistant_content = response['message']['content']

            # Add assistant message to history (only if using instance messages)
            if messages is None:
                conversation_messages.append({"role": "assistant", "content": assistant_content})
                self.messages = conversation_messages

            print(Fore.GREEN + f"Response generated in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
            logger.info("Response generated successfully")

            return assistant_content

        return self.service_manager.execute_sync('llm_generation', _generate_operation)
