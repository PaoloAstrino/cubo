from sentence_transformers import SentenceTransformer
import torch
from colorama import Fore, Style
from src.config import config
from src.logger import logger

class ModelManager:
    """Manages the loading and configuration of CUBO's embedding model."""

    def __init__(self):
        self.model = None
        self.device = self._detect_device()

    def _detect_device(self) -> str:
        """Detect available device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("CUDA not available. Using CPU.")
        return device

    def load_model(self):
        """Load the embedding model with GPU fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model...")
            import time
            start = time.time()

            self.model = SentenceTransformer(config.get("model_path"), device=self.device)

            logger.info(f"Model loaded in {time.time() - start:.2f} seconds.")
            logger.info("Embedding model loaded successfully.")
            return self.model
        except Exception as e:
            if self.device == 'cuda':
                logger.warning(f"GPU loading failed ({e}). Falling back to CPU...")
                logger.warning(f"GPU loading failed: {e}. Retrying with CPU.")
                self.device = 'cpu'
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(config.get("model_path"), device=self.device)
                    logger.info(f"Model loaded on CPU in {time.time() - start:.2f} seconds.")
                    logger.info("Embedding model loaded on CPU after GPU failure.")
                    return self.model
                except Exception as e2:
                    logger.error(f"Failed to load model on CPU: {e2}")
                    logger.error(f"Error loading model on CPU: {e2}")
                    raise
            else:
                logger.error(f"Failed to load model: {e}")
                logger.error(f"Error loading model: {e}")
                raise

    def get_model(self):
        """Get the loaded model, loading it if necessary."""
        if self.model is None:
            self.load_model()
        return self.model

# Global model manager instance
model_manager = ModelManager()
