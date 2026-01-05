"""Utility wrapper around SentenceTransformers embedding generation."""

from typing import List, Optional

from cubo.config import config
from cubo.embeddings.model_inference_threading import get_model_inference_threading
from cubo.embeddings.model_loader import model_manager
from cubo.utils.logger import logger


class EmbeddingGenerator:
    """Encapsulates SentenceTransformer encoding with centralized batching and logging.

    Adds optional support for instruction-tuned embedding models that require
    text prefixes (prompts) for different modes (e.g. `query` vs `document`).

    If the underlying model directory contains `config_sentence_transformers.json`
    with a `prompts` mapping, the generator can automatically prefix texts
    when `prompt_name` is provided to `encode`.
    """

    def __init__(self, model=None, batch_size: Optional[int] = None, inference_threading=None):
        self.model = model or model_manager.get_model()
        self.batch_size = batch_size or config.get("embedding_batch_size", 32)
        self._threading = inference_threading or get_model_inference_threading()

        # Try to load prompt mappings from model config (if present)
        self._prompts = self._load_prompts_from_model_path(self.model_path)

    @property
    def model_path(self) -> Optional[str]:
        """Return the model path used by the underlying model manager."""
        return getattr(model_manager, "_model_path", None) or config.get("model_path")

    @staticmethod
    def _load_prompts_from_model_path(model_path: Optional[str]) -> dict:
        """Load `prompts` mapping from `config_sentence_transformers.json` in model dir.

        Returns an empty dict if no prompts config is available.
        """
        if not model_path:
            return {}
        try:
            import json
            from pathlib import Path

            cfg_path = Path(model_path) / "config_sentence_transformers.json"
            if not cfg_path.exists():
                return {}
            with open(cfg_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            prompts = payload.get("prompts") or {}
            return prompts
        except Exception:
            return {}

    @classmethod
    def get_prompt_prefix_for_model(
        cls, model_path: Optional[str], prompt_name: str
    ) -> Optional[str]:
        """Get a prompt prefix for a given model path and prompt name.

        Tries multiple fallbacks to match common prompt key variants.
        """
        prompts = cls._load_prompts_from_model_path(model_path)
        if not prompts:
            return None
        # Exact match
        if prompt_name in prompts:
            return prompts.get(prompt_name)
        # Common fallbacks
        alt_keys = [
            prompt_name.lower(),
            prompt_name.capitalize(),
            f"Retrieval-{prompt_name}",
            f"Retrieval-{prompt_name.capitalize()}",
            f"Retrieval-{prompt_name.lower()}",
            f"{prompt_name}-document",
            f"{prompt_name}-query",
        ]
        for k in alt_keys:
            if k in prompts:
                return prompts.get(k)
        # Last resort: search for any key containing the prompt_name substring
        for k, v in prompts.items():
            if prompt_name.lower() in k.lower():
                return v
        return None

    def _apply_prompt_to_texts(self, texts: List[str], prompt_name: Optional[str]) -> List[str]:
        """Return texts prefixed with the model prompt if available."""
        if not prompt_name:
            return texts
        prefix = None
        try:
            prefix = self.get_prompt_prefix_for_model(self.model_path, prompt_name)
        except Exception:
            prefix = None
        if not prefix:
            # No prefix found; do not alter texts
            logger.debug(
                f"No prompt prefix found for model {self.model_path} and prompt '{prompt_name}'"
            )
            return texts
        return [prefix + t for t in texts]

    def encode(
        self, texts: List[str], batch_size: Optional[int] = None, prompt_name: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed
            batch_size: Optional override for batch size
            prompt_name: Optional prompt name to select a prefix (e.g. 'query' or 'document')
        """
        if not texts:
            return []
        batch_size = batch_size or self.batch_size
        texts_to_encode = self._apply_prompt_to_texts(texts, prompt_name)

        logger.info(
            f"EmbeddingGenerator encoding {len(texts)} texts (batch_size={batch_size}, prompt_name={prompt_name})"
        )
        embeddings = self._threading.generate_embeddings_threaded(
            texts_to_encode, self.model, batch_size=batch_size
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
