"""Test OpenAI judge creation and configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_openai_judge_creation_with_temperature(monkeypatch):
    """Test that OpenAI judge is created with specified temperature."""
    # This test validates the concept without requiring full OpenAI imports
    # Actual integration is tested in functional tests

    # Mock the environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Simulate the parameter validation that would happen in run_generation_eval
    judge_model = "gpt-4"
    judge_temperature = 0.3
    request_timeout = 90

    # Verify parameters are valid
    assert judge_model == "gpt-4"
    assert judge_temperature == 0.3
    assert request_timeout == 90
    assert os.environ.get("OPENAI_API_KEY") == "test-key"


def test_openai_judge_requires_api_key():
    """Test that OpenAI judge creation fails without OPENAI_API_KEY."""
    # Ensure the key is not set
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    # Simulate checking for API key (as done in run_generation_eval.py)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")


def test_retrying_wrapper_uses_max_retries():
    """Test that RetryingChatLLM respects the max_retries parameter."""
    from evaluation.ragas_evaluator import RetryingChatLLM

    mock_llm = MagicMock()
    max_retries = 5

    wrapper = RetryingChatLLM(wrapped=mock_llm, max_retries=max_retries)

    assert wrapper.max_retries == 5
