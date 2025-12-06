from typing import List, Dict, Optional

class ChatTemplateManager:
    """Manages chat templates for different LLMs to ensure correct prompt formatting."""

    def __init__(self):
        self.templates = {
            "llama3": self._format_llama3,
            "default": self._format_default
        }

    def format_chat(self, messages: List[Dict[str, str]], model_name: Optional[str] = None) -> str:
        """Formats a list of messages into a single string based on the model's template.

        Args:
            messages: A list of dictionaries with 'role' and 'content'.
            model_name: The name of the model to use for formatting.

        Returns:
            A formatted prompt string.
        """
        # specialized logic to detect llama 3 variations
        if model_name:
            model_lower = model_name.lower()
            if "llama-3" in model_lower or "llama3" in model_lower:
                return self._format_llama3(messages)
        
        # Fallback to default
        return self._format_default(messages)

    def format_user_message(self, context: str, query: str) -> str:
        """Format the user message content (context + question).
        
        This provides a consistent format for the user message content
        that works well with both Ollama and local llama_cpp backends.
        
        Args:
            context: Document context to include
            query: User's question
            
        Returns:
            Formatted user message content
        """
        return f"Context:\n{context}\n\nQuestion: {query}"

    def _format_llama3(self, messages: List[Dict[str, str]]) -> str:
        """Formats messages using Llama 3 chat template.
        
        Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {assistant_prompt}<|eot_id|>...
        """
        formatted_text = "<|begin_of_text|>"
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            formatted_text += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            formatted_text += f"{content}<|eot_id|>"
        
        # If the last message is from user, prepare for assistant generation
        if messages and messages[-1]["role"] == "user":
            formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted_text

    def _format_default(self, messages: List[Dict[str, str]]) -> str:
        """Default formatting for generic models."""
        formatted_text = ""
        for message in messages:
            role = message.get("role", "").capitalize()
            content = message.get("content", "")
            formatted_text += f"{role}: {content}\n\n"
        
        formatted_text = formatted_text.strip()
        
        # Prepare for assistant
        if messages and messages[-1]["role"] == "user":
            formatted_text += "\n\nAssistant: "
            
        return formatted_text
