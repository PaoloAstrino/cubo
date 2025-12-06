import unittest
from cubo.processing.chat_template_manager import ChatTemplateManager

class TestChatTemplateManager(unittest.TestCase):
    def setUp(self):
        self.manager = ChatTemplateManager()

    def test_llama3_formatting(self):
        messages = [
            {"role": "system", "content": "System Prompt"},
            {"role": "user", "content": "User Query"}
        ]
        expected = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\nSystem Prompt<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nUser Query<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        result = self.manager.format_chat(messages, model_name="llama-3-8b-instruct")
        self.assertEqual(result, expected)

    def test_default_formatting(self):
        messages = [
            {"role": "system", "content": "System Prompt"},
            {"role": "user", "content": "User Query"}
        ]
        expected = (
            "System: System Prompt\n\n"
            "User: User Query\n\n"
            "Assistant: "
        )
        result = self.manager.format_chat(messages, model_name="unknown-model")
        self.assertEqual(result, expected)

    def test_llama3_case_insensitivity(self):
        messages = [{"role": "user", "content": "Hi"}]
        result_upper = self.manager.format_chat(messages, model_name="LLAMA3")
        self.assertTrue("<|start_header_id|>" in result_upper)

    def test_llama3_with_history(self):
        """Test that conversation history is formatted correctly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = self.manager.format_chat(messages, model_name="llama3")
        # Check all parts are present
        self.assertIn("<|begin_of_text|>", result)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", result)
        self.assertIn("You are helpful.", result)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", result)
        self.assertIn("Hello", result)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", result)
        self.assertIn("Hi there!", result)
        self.assertIn("How are you?", result)
        # Should end with assistant header ready for generation
        self.assertTrue(result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))

if __name__ == '__main__':
    unittest.main()
