import json
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from evaluation.ragas_evaluator import RetryingChatLLM


class FakeLLM:
    """A fake LLM that returns non-JSON on first call and valid JSON on second call."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages=None, stop=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            # Non-JSON text
            text = "This is not valid JSON. Sorry!"
        else:
            # Valid JSON text
            text = json.dumps({"result": "ok", "score": 0.9})

        # Return a ChatResult-like object
        gen = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[gen])


def test_retrying_chat_llm_retries_and_writes_debug(tmp_path):
    fake = FakeLLM()
    debug_dir = str(tmp_path / "retry_debug")

    wrapper = RetryingChatLLM(wrapped=fake, max_retries=2, debug_dir=debug_dir)

    # Call wrapper with a simple message
    messages = [HumanMessage(content="Please respond in JSON only.")]

    res = wrapper._generate(messages=messages)

    # After retries, we should get a ChatResult back
    assert hasattr(res, "generations")
    assert len(res.generations) > 0

    # Ensure the fake LLM was called more than once (i.e., a retry occurred)
    assert fake.calls >= 2

    # Check debug files created (initial attempt + at least one retry attempt)
    files = list(Path(debug_dir).glob("call_*.txt"))
    assert len(files) >= 2

    # Also verify that a failure summary file is not present (since retry succeeded)
    failed_files = list(Path(debug_dir).glob("*_FAILED.json"))
    assert len(failed_files) == 0
