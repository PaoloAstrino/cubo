"""Tests for system prompt centralization and enforcement.

Ensures that both ResponseGenerator and LocalResponseGenerator use the
canonical DEFAULT_SYSTEM_PROMPT when no config override is provided.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from cubo.config import config
from cubo.config.prompt_defaults import DEFAULT_SYSTEM_PROMPT
from cubo.processing.generator import ResponseGenerator


class TestSystemPromptCentralization:
    """Test suite for system prompt centralization."""

    def test_canonical_prompt_used_by_response_generator(self):
        """ResponseGenerator should use DEFAULT_SYSTEM_PROMPT when config key is not set."""
        # Clear config key to test fallback
        original_value = config.get("llm.system_prompt")
        try:
            config.set("llm.system_prompt", None)
            
            generator = ResponseGenerator()
            generator.initialize_conversation()
            
            # Assert system message uses canonical prompt
            assert generator.messages[0]["role"] == "system"
            assert generator.messages[0]["content"] == DEFAULT_SYSTEM_PROMPT
            assert "cite sources" in generator.messages[0]["content"].lower()
            assert "not in provided context" in generator.messages[0]["content"].lower()
        finally:
            # Restore original config
            if original_value:
                config.set("llm.system_prompt", original_value)

    @patch("cubo.processing.llm_local.Llama")
    def test_local_response_generator_uses_canonical_prompt_when_no_messages(self, mock_llama):
        """LocalResponseGenerator should use DEFAULT_SYSTEM_PROMPT for system message."""
        from cubo.processing.llm_local import LocalResponseGenerator
        
        # Mock the Llama model initialization
        mock_model_instance = MagicMock()
        mock_model_instance.create_completion.return_value = {"choices": [{"text": "test response"}]}
        mock_llama.return_value = mock_model_instance
        
        original_value = config.get("llm.system_prompt")
        original_path = config.get("local_llama_model_path")
        
        try:
            config.set("llm.system_prompt", None)
            config.set("local_llama_model_path", "./test_model.gguf")
            
            # Create generator with mocked model
            generator = LocalResponseGenerator()
            generator._llm = mock_model_instance
            
            # Mock the service manager
            with patch.object(generator.service_manager, 'execute_sync', side_effect=lambda name, fn: fn()):
                response = generator.generate_response("test query", "test context", messages=None)
            
            # Verify the call - should contain system message with canonical prompt
            call_args = mock_model_instance.create_completion.call_args
            prompt_arg = call_args.kwargs.get("prompt") or call_args.args[0] if call_args.args else None
            
            assert prompt_arg is not None
            # The formatted prompt should contain the canonical system message
            assert "cite sources" in prompt_arg.lower() or "Source" in prompt_arg
            assert "not in provided context" in prompt_arg.lower() or "Not in" in prompt_arg
        finally:
            if original_value:
                config.set("llm.system_prompt", original_value)
            if original_path:
                config.set("local_llama_model_path", original_path)
            else:
                config.set("local_llama_model_path", None)

    def test_config_override_applies_to_both_generators(self):
        """Custom config value should override DEFAULT_SYSTEM_PROMPT for both generators."""
        custom_prompt = "Custom system prompt for testing"
        original_value = config.get("llm.system_prompt")
        
        try:
            config.set("llm.system_prompt", custom_prompt)
            
            # Test ResponseGenerator
            generator = ResponseGenerator()
            assert generator.system_prompt == custom_prompt
            generator.initialize_conversation()
            assert generator.messages[0]["content"] == custom_prompt
            
        finally:
            if original_value:
                config.set("llm.system_prompt", original_value)
            else:
                config.set("llm.system_prompt", None)

    def test_default_prompt_contains_citation_requirements(self):
        """DEFAULT_SYSTEM_PROMPT should include citation and 'not in context' instructions."""
        assert "cite" in DEFAULT_SYSTEM_PROMPT.lower() or "Source" in DEFAULT_SYSTEM_PROMPT
        assert "not in" in DEFAULT_SYSTEM_PROMPT.lower()
        assert "provided context" in DEFAULT_SYSTEM_PROMPT.lower()
        
    def test_default_prompt_discourages_hallucination(self):
        """DEFAULT_SYSTEM_PROMPT should discourage external knowledge and invention."""
        prompt_lower = DEFAULT_SYSTEM_PROMPT.lower()
        # Check for hallucination mitigation phrases
        assert any(phrase in prompt_lower for phrase in [
            "only the provided context",
            "do not use external knowledge",
            "use only the provided context"
        ])

    @patch("cubo.processing.generator.ollama")
    def test_provider_switch_respects_canonical_prompt(self, mock_ollama):
        """Switching providers should still respect the canonical system prompt."""
        original_value = config.get("llm.system_prompt")
        
        try:
            config.set("llm.system_prompt", None)
            
            # Test with Ollama provider
            mock_ollama.chat.return_value = {"message": {"content": "test response"}}
            generator = ResponseGenerator()
            assert generator.system_prompt == DEFAULT_SYSTEM_PROMPT
            
        finally:
            if original_value:
                config.set("llm.system_prompt", original_value)
