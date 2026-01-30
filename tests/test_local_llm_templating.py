import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cubo.processing.llm_local import LocalResponseGenerator
from cubo.config import config


class TestLocalLLMFormatting(unittest.TestCase):
    """
    Tests the prompt formatting logic in LocalResponseGenerator
    to ensure it correctly uses ChatTemplateManager.
    """

    def setUp(self):
        # Mock llama_cpp module entirely before it's imported anywhere
        self.mock_llama_mod = MagicMock()
        sys.modules["llama_cpp"] = self.mock_llama_mod

        # Setup the mock Llama class
        self.mock_llama_inst = MagicMock()
        self.mock_llama_mod.Llama.return_value = self.mock_llama_inst

        # Mock config values - use a generic path to avoid string matching hits
        config.set("local_llama_model_path", "models/test-model.gguf")
        config.set("llm_model", "llama3")

        # Mock create_completion return value
        self.mock_llama_inst.create_completion.return_value = {
            "choices": [{"text": "Mocked Answer"}]
        }

    def tearDown(self):
        # Clean up sys.modules
        if "llama_cpp" in sys.modules:
            del sys.modules["llama_cpp"]
        # Clear config keys
        config.set("llm_model", None)
        config.set("local_llama_model_path", None)

    def test_llama3_template_application(self):
        """
        Verify that Llama 3 tags are present in the final prompt string.
        """
        generator = LocalResponseGenerator()

        query = "What is CUBO?"
        context = "CUBO is a local RAG system."

        generator.generate_response(query, context)

        # Inspect the prompt passed to create_completion
        args, kwargs = self.mock_llama_inst.create_completion.call_args
        prompt = kwargs.get("prompt", args[0] if args else "")

        # Check for Llama 3 specific tags
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", prompt)
        self.assertIn("CUBO is a local RAG system", prompt)
        self.assertIn("What is CUBO?", prompt)

    def test_mistral_template_application(self):
        """
        Verify that Mistral tags are present when model is mistral.
        """
        config.set("llm_model", "mistral")
        generator = LocalResponseGenerator()

        query = "Hello"
        context = "Context info"

        generator.generate_response(query, context)

        args, kwargs = self.mock_llama_inst.create_completion.call_args
        prompt = kwargs.get("prompt", args[0] if args else "")

        # Check for Mistral specific tags
        self.assertIn("<s>", prompt)
        self.assertIn("[INST]", prompt)
        self.assertIn("[/INST]", prompt)


if __name__ == "__main__":
    unittest.main()
