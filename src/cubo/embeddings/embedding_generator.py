"""Utility wrapper around SentenceTransformers embedding generation."""

from typing import List, Optional

from src.cubo.config import config
from src.cubo.embeddings.model_inference_threading import get_model_inference_threading
from src.cubo.embeddings.model_loader import model_manager
from src.cubo.utils.logger import logger


class EmbeddingGenerator:
    """Encapsulates SentenceTransformer encoding with centralized batching and logging."""

    def __init__(self, model=None, batch_size: Optional[int] = None, inference_threading=None):
        self.model = model or model_manager.get_model()
        self.batch_size = batch_size or config.get("embedding_batch_size", 32)
        self._threading = inference_threading or get_model_inference_threading()

    def encode(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        batch_size = batch_size or self.batch_size
        logger.info(f"EmbeddingGenerator encoding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self._threading.generate_embeddings_threaded(
            texts, self.model, batch_size=batch_size
        )
        return embeddings

    def embed_chunks(
        self, df_rows: List[str], text_column: str = "text", batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Embed a list of chunk rows (or list of strings) using the configured model.

        Supports being passed either a list of strings or a list-like of dict/records where the
        text content is available under `text_column`.
        """
        # Allow list of dataframes or dicts: extract text
        if not df_rows:
            return []
        if isinstance(df_rows[0], dict):
            texts = [row.get(text_column, "") for row in df_rows]
        else:
            texts = [str(x) for x in df_rows]
        return self.encode(texts, batch_size=batch_size)

    def embed_summaries(
        self, df_rows: List[str], summary_column: str = "summary", batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Embed summaries from a list of records or strings.

        If df_rows is a list of dict-like records, uses `summary_column`; otherwise it treats
        the list as strings to encode directly.
        """
        if not df_rows:
            return []
        if isinstance(df_rows[0], dict):
            texts = [row.get(summary_column, "") for row in df_rows]
        else:
            texts = [str(x) for x in df_rows]
        return self.encode(texts, batch_size=batch_size)
