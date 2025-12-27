import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock llama_cpp module for environments where it's not installed
sys.modules["llama_cpp"] = MagicMock()

from cubo.config import config
from cubo.processing.chat_template_manager import ChatTemplateManager
from cubo.processing.llm_local import LocalResponseGenerator


class TestLLMModernization(unittest.TestCase):
    def setUp(self):
        self.manager = ChatTemplateManager()

    def test_mistral_template_detection(self):
        messages = [{"role": "user", "content": "Hello"}]
        # Should detect Mistral
        prompt = self.manager.format_chat(messages, model_name="Mistral-7B-Instruct")
        self.assertIn("[INST] Hello [/INST]", prompt)
        self.assertIn("<s>", prompt)

    def test_chatml_template_detection_qwen(self):
        messages = [{"role": "user", "content": "Hello"}]
        # Should detect ChatML for Qwen
        prompt = self.manager.format_chat(messages, model_name="Qwen-14B-Chat")
        self.assertIn("<|im_start|>user\nHello<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_chatml_template_detection_yi(self):
        messages = [{"role": "user", "content": "Hello"}]
        # Should detect ChatML for Yi
        prompt = self.manager.format_chat(messages, model_name="Yi-34B-Chat")
        self.assertIn("<|im_start|>user\nHello<|im_end|>", prompt)

    @patch("llama_cpp.Llama")
    def test_local_llm_context_config(self, MockLlama):
        # Setup mock
        MockLlama.return_value = MagicMock()

        # 1. Test Default (should be 0 or 8192 depending on how we set the fallback)
        # We manually set config values using the update method or overrides if possible
        # But here we can use patch.dict on config._settings if it was a dict, but it's an object.
        # Let's rely on the config.set method.

        original_n_ctx = config.get("llm.n_ctx")

        try:
            # Case A: Config set to 4096
            config.set("llm.n_ctx", 4096)
            config.set(
                "local_llama_model_path", "dummy/path"
            )  # Ensure init doesn't fail on path check

            gen = LocalResponseGenerator(model_path="dummy/path")

            # Verify Llama was called with n_ctx=4096
            MockLlama.assert_called_with(model_path="dummy/path", n_ctx=4096, n_gpu_layers=0)

            # Case B: Config not set (None/0) -> Should default to 8192 manually in our code logic ONLY IF it was missing.
            # But we set 0 in __init__.py default.
            config.set("llm.n_ctx", 0)
            gen = LocalResponseGenerator(model_path="dummy/path")

            # The logic we wrote: n_ctx = config.get("llm.n_ctx", 0). If n_ctx is None: fallback.
            # Since we set 0, it should pass 0 (auto).
            MockLlama.assert_called_with(model_path="dummy/path", n_ctx=0, n_gpu_layers=0)

        finally:
            # Cleanup
            config.set("llm.n_ctx", original_n_ctx)


if __name__ == "__main__":
    unittest.main()
