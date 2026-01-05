from cubo.config import config
from cubo.embeddings.embedding_generator import EmbeddingGenerator
from cubo.retrieval.retrieval_executor import RetrievalExecutor


class DummyThreading:
    def __init__(self):
        self.last_texts = None

    def generate_embeddings_threaded(self, texts, model, batch_size=8):
        self.last_texts = texts
        return [[0.0] * 4 for _ in texts]


class DummyModel:
    pass


def test_generate_query_embedding_applies_prompt():
    # Ensure model path points to embeddinggemma model which has a query prompt
    config.set("model_path", "./models/embeddinggemma-300m")
    dt = DummyThreading()
    re = RetrievalExecutor(
        collection=None, bm25_searcher=None, model=DummyModel(), inference_threading=dt
    )
    emb = re.generate_query_embedding("hello")
    assert dt.last_texts is not None
    # The captured text should have the query prompt prefix
    assert dt.last_texts[0].startswith("task:") or "query" in dt.last_texts[0]
    assert emb == [0.0] * 4
