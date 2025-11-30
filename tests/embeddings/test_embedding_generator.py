from unittest.mock import MagicMock, patch

from cubo.embeddings.embedding_generator import EmbeddingGenerator


def test_embedding_generator_uses_threading():
    fake_threading = MagicMock()
    fake_threading.generate_embeddings_threaded.return_value = [[0.1, 0.2]]
    fake_model = MagicMock()

    with patch(
        "src.cubo.embeddings.embedding_generator.get_model_inference_threading",
        return_value=fake_threading,
    ):
        with patch("src.cubo.embeddings.embedding_generator.model_manager") as mock_manager:
            mock_manager.get_model.return_value = fake_model
            generator = EmbeddingGenerator(batch_size=16)
            embeddings = generator.encode(["hello world"])
            assert embeddings == [[0.1, 0.2]]
            fake_threading.generate_embeddings_threaded.assert_called_once_with(
                ["hello world"], fake_model, batch_size=16
            )


def test_embedding_generator_embed_summaries_and_chunks():
    fake_threading = MagicMock()
    fake_threading.generate_embeddings_threaded.return_value = [[0.2, 0.3]]
    fake_model = MagicMock()
    with patch(
        "src.cubo.embeddings.embedding_generator.get_model_inference_threading",
        return_value=fake_threading,
    ):
        with patch("src.cubo.embeddings.embedding_generator.model_manager") as mock_manager:
            mock_manager.get_model.return_value = fake_model
            generator = EmbeddingGenerator(batch_size=8)
            # Test embed_chunks with plain list
            chunk_embeddings = generator.embed_chunks(["chunk1 text"])
            assert chunk_embeddings == [[0.2, 0.3]]
            # Test embed_summaries with records list
            summaries = [{"summary": "short summary"}]
            summary_embeddings = generator.embed_summaries(summaries)
            assert summary_embeddings == [[0.2, 0.3]]
