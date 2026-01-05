import os
from cubo.embeddings.embedding_generator import EmbeddingGenerator


def test_get_prompt_prefix_from_model_dir():
    model_path = os.path.join(os.getcwd(), "models", "embeddinggemma-300m")
    prefix = EmbeddingGenerator.get_prompt_prefix_for_model(model_path, "query")
    assert prefix is not None
    assert "query" in prefix or "task" in prefix


class DummyThreading:
    def __init__(self):
        self.last_texts = None

    def generate_embeddings_threaded(self, texts, model, batch_size=8):
        # capture for assertion and return a dummy embedding
        self.last_texts = texts
        return [[0.0] * 4 for _ in texts]


def test_encode_applies_prompt_prefix():
    model_path = os.path.join(os.getcwd(), "models", "embeddinggemma-300m")
    gen = EmbeddingGenerator(batch_size=2, inference_threading=DummyThreading())
    # Override model_path to known model with prompts
    gen._prompts = gen._load_prompts_from_model_path(model_path)

    texts = ["hello world"]
    _ = gen.encode(texts, batch_size=2, prompt_name="query")

    # Ensure the threading layer saw a prefixed string
    seen = gen._threading.last_texts
    assert seen is not None
    assert seen[0].startswith("task:") or seen[0].startswith("query:") or "query" in seen[0]
