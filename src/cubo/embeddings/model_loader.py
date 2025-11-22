import torch

from src.cubo.config import config
from src.cubo.utils.logger import logger


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
            start_time = self._start_loading_timer()
            self.model = self._load_model_on_device(self.device)
            self._log_successful_loading(start_time)
            return self.model
        except Exception as e:
            return self._handle_loading_failure(e, start_time)

    def _start_loading_timer(self):
        """Start timing the model loading process."""
        import time
        logger.info("Loading embedding model...")
        return time.time()

    def _load_model_on_device(self, device: str):
        """Load the SentenceTransformer model on the specified device."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(config.get("model_path"), device=device)

    def _log_successful_loading(self, start_time: float):
        """Log successful model loading with timing."""
        import time
        duration = time.time() - start_time
        logger.info(f"Model loaded in {duration:.2f} seconds.")
        logger.info("Embedding model loaded successfully.")

    def _handle_loading_failure(self, initial_error: Exception, start_time: float):
        """Handle model loading failure with GPU-to-CPU fallback."""
        if self.device == 'cuda':
            return self._fallback_to_cpu_loading(initial_error, start_time)
        else:
            logger.error(f"Failed to load model: {initial_error}")
            logger.error(f"Error loading model: {initial_error}")
            raise

    def _fallback_to_cpu_loading(self, gpu_error: Exception, start_time: float):
        """Fallback to CPU loading when GPU fails."""
        logger.warning(f"GPU loading failed ({gpu_error}). Falling back to CPU...")
        logger.warning(f"GPU loading failed: {gpu_error}. Retrying with CPU.")
        self.device = 'cpu'

        try:
            self.model = self._load_model_on_device(self.device)
            self._log_cpu_fallback_success(start_time)
            return self.model
        except Exception as cpu_error:
            logger.error(f"Failed to load model on CPU: {cpu_error}")
            logger.error(f"Error loading model on CPU: {cpu_error}")
            raise

    def _log_cpu_fallback_success(self, start_time: float):
        """Log successful CPU loading after GPU fallback."""
        import time
        duration = time.time() - start_time
        logger.info(f"Model loaded on CPU in {duration:.2f} seconds.")
        logger.info("Embedding model loaded on CPU after GPU failure.")

    def get_model(self):
        """Get the loaded model, loading it if necessary."""
        if self.model is None:
            self.load_model()
        return self.model


# Global model manager instance
model_manager = ModelManager()
