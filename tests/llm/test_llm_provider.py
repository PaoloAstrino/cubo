from src.cubo.config import config
from src.cubo.processing.generator import ResponseGenerator, create_response_generator


def test_create_response_generator_default():
    config.set("llm.provider", "ollama")
    gen = create_response_generator()
    assert isinstance(gen, ResponseGenerator)


def test_create_response_generator_local():
    # If local provider not installed, factory should fall back to default ResponseGenerator
    config.set("llm.provider", "local")
    gen = create_response_generator()
    assert hasattr(gen, "generate_response")
