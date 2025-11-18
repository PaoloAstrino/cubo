"""Utility wrapper around SentenceTransformers embedding generation."""
from typing import List, Optional

from src.cubo.config import config
from src.cubo.utils.logger import logger
from src.cubo.embeddings.model_loader import model_manager
from src.cubo.embeddings.model_inference_threading import get_model_inference_threading


class EmbeddingGenerator:
    """Encapsulates SentenceTransformer encoding with centralized batching and logging."""

    def __init__(
        self,
        model=None,
        batch_size: Optional[int] = None,
        inference_threading=None
    ):
        self.model = model or model_manager.get_model()
        self.batch_size = batch_size or config.get('embedding_batch_size', 32)
        self._threading = inference_threading or get_model_inference_threading()

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        batch_size = batch_size or self.batch_size
        logger.info(f"EmbeddingGenerator encoding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self._threading.generate_embeddings_threaded(texts, self.model, batch_size=batch_size)
        return embeddings
