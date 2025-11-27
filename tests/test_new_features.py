"""Integration test for new features."""

import pytest

from src.cubo.config import config
from src.cubo.processing.generator import ResponseGenerator


def test_llm_config_system_prompt():
    """Test that system prompt is configurable."""
    # Set custom system prompt
    custom_prompt = "You are a test assistant."
    config.set("llm.system_prompt", custom_prompt)

    generator = ResponseGenerator()

    assert generator.system_prompt == custom_prompt

    # Reset to default
    config.set("llm.system_prompt", None)


def test_llm_config_model_name():
    """Test that model name fallback works."""
    # Test primary config key
    config.set("llm.model_name", "test-model")
    generator = ResponseGenerator()

    # The model name should be retrievable
    model_name = config.get("llm.model_name")
    assert model_name == "test-model"

    # Reset
    config.set("llm.model_name", None)


def test_scaffold_config():
    """Test scaffold configuration."""
    # Test default values
    use_clustering = config.get("scaffold.use_semantic_clustering", False)
    clustering_method = config.get("scaffold.clustering_method", "kmeans")
    scaffold_size = config.get("scaffold.scaffold_size", 5)

    assert isinstance(use_clustering, bool)
    assert clustering_method in ["kmeans", "hdbscan"]
    assert scaffold_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
