import torch
from sentence_transformers import SentenceTransformer
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
            print(Fore.GREEN + f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}" + Style.RESET_ALL)
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print(Fore.YELLOW + "CUDA not available. Using CPU." + Style.RESET_ALL)
            logger.info("Using CPU (CUDA not available)")
        return device

    def load_model(self) -> SentenceTransformer:
        """Load the embedding model with GPU fallback."""
        try:
            print(Fore.BLUE + "Loading embedding model..." + Style.RESET_ALL)
            import time
            start = time.time()

            self.model = SentenceTransformer(config.get("model_path"), device=self.device)

            print(Fore.GREEN + f"Model loaded in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
            logger.info("Embedding model loaded successfully.")
            return self.model
        except Exception as e:
            if self.device == 'cuda':
                print(Fore.YELLOW + f"GPU loading failed ({e}). Falling back to CPU..." + Style.RESET_ALL)
                logger.warning(f"GPU loading failed: {e}. Retrying with CPU.")
                self.device = 'cpu'
                try:
                    self.model = SentenceTransformer(config.get("model_path"), device=self.device)
                    print(Fore.GREEN + f"Model loaded on CPU in {time.time() - start:.2f} seconds." + Style.RESET_ALL)
                    logger.info("Embedding model loaded on CPU after GPU failure.")
                    return self.model
                except Exception as e2:
                    logger.error(f"Failed to load model on CPU: {e2}")
                    print(Fore.RED + f"Error loading model on CPU: {e2}" + Style.RESET_ALL)
                    raise
            else:
                logger.error(f"Failed to load model: {e}")
                print(Fore.RED + f"Error loading model: {e}" + Style.RESET_ALL)
                raise

    def get_model(self) -> SentenceTransformer:
        """Get the loaded model, loading it if necessary."""
        if self.model is None:
            self.load_model()
        return self.model

# Global model manager instance
model_manager = ModelManager()
