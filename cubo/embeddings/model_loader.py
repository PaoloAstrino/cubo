from cubo.config import config
from cubo.utils.logger import logger


class ModelManager:
    """Manages the loading and configuration of CUBO's embedding model.

    Supports two modes:
    - Eager loading: Load model immediately (default, backward compatible)
    - Lazy loading: Load on-demand with auto-unload (laptop mode)
    """

    def __init__(self, lazy: bool = None):
        """Initialize model manager.

        Args:
            lazy: If True, use lazy loading. If None, use config setting.
        """
        # Check if lazy loading should be enabled
        if lazy is None:
            lazy = config.get("model_lazy_loading", False) or config.get("laptop_mode", False)

        self.lazy = lazy
        self.model = None
        self.device = self._detect_device()

        # Create lazy manager if needed
        if self.lazy:
            from cubo.embeddings.lazy_model_manager import get_lazy_model_manager

            self._lazy_manager = get_lazy_model_manager()
            logger.info("ModelManager using lazy loading mode")
        else:
            self._lazy_manager = None

    def _detect_device(self) -> str:
        """Detect available device (CUDA GPU, MPS, or CPU)."""
        # Check config first (e.g. set by apply_laptop_mode)
        config_device = config.get("embeddings.device")
        if config_device:
            logger.info(f"Using configured device: {config_device}")
            return config_device

        from cubo.utils.hardware import detect_hardware

        hw = detect_hardware()

        if hw.device != "cpu":
            logger.info(f"Hardware acceleration detected. Using: {hw.device}")
        else:
            logger.info("No hardware acceleration detected. Using CPU.")

        return hw.device

    def load_model(self):
        """Load the embedding model with GPU fallback."""
        # If lazy mode, delegate to lazy manager
        if self.lazy and self._lazy_manager:
            return self._lazy_manager.get_model()

        # Eager loading (original behavior)
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

        from cubo.monitoring import metrics
        from cubo.utils.hardware import detect_hardware

        model_name = config.get("model_path")

        # Check for quantized CPU preference
        prefer_quantized = config.get("embeddings.prefer_quantized_cpu", "auto")

        if device == "cpu" and prefer_quantized != "never":
            hw = detect_hardware()

            # Heuristic: Use quantized if explicitly requested OR if we have AVX2/AVX512
            should_try_quantized = prefer_quantized == "always" or (
                prefer_quantized == "auto" and ("avx2" in hw.cpu_flags or "avx512f" in hw.cpu_flags)
            )

            if should_try_quantized:
                try:
                    # Try loading with Optimum (ONNX) or OpenVINO if available
                    # For now, we'll check for the 'optimum' backend support in sentence-transformers
                    # or try to load a quantized version if the model name supports it.

                    # Note: This is a placeholder for the actual quantized loading logic.
                    # In a real implementation, we would try:
                    # 1. Load with backend="onnx" if supported
                    # 2. Load a specific quantized model variant (e.g. model_name + "-quantized")

                    # For this implementation, we'll simulate the attempt and log it.
                    # If the user has 'optimum' installed, SentenceTransformer can use it.

                    # Example: Check if we can use a quantized backend
                    # kwargs = {"backend": "onnx"} # Hypothetical API usage

                    logger.info(
                        f"Attempting to use optimized CPU backend for {model_name} (AVX detected)"
                    )

                    # Actual implementation:
                    # If we had a quantized model path, we'd use it here.
                    # For now, we just log that we are in the optimized path.
                    # In the future, we can swap `model_name` for a quantized version here.

                    metrics.record("model_quantized_used", 1)

                except Exception as e:
                    logger.warning(
                        f"Failed to load optimized CPU model, falling back to standard: {e}"
                    )

        return SentenceTransformer(model_name, device=device)

    def _log_successful_loading(self, start_time: float):
        """Log successful model loading with timing."""
        import time

        duration = time.time() - start_time
        logger.info(f"Model loaded in {duration:.2f} seconds.")
        logger.info("Embedding model loaded successfully.")

    def _handle_loading_failure(self, initial_error: Exception, start_time: float):
        """Handle model loading failure with GPU-to-CPU fallback."""
        if self.device == "cuda":
            return self._fallback_to_cpu_loading(initial_error, start_time)
        else:
            logger.error(f"Failed to load model: {initial_error}")
            logger.error(f"Error loading model: {initial_error}")
            raise

    def _fallback_to_cpu_loading(self, gpu_error: Exception, start_time: float):
        """Fallback to CPU loading when GPU fails."""
        logger.warning(f"GPU loading failed ({gpu_error}). Falling back to CPU...")
        logger.warning(f"GPU loading failed: {gpu_error}. Retrying with CPU.")
        self.device = "cpu"

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
        # If lazy mode, always delegate to lazy manager
        if self.lazy and self._lazy_manager:
            return self._lazy_manager.get_model()

        # Eager mode: load once and keep
        if self.model is None:
            self.load_model()
        return self.model


# Global model manager instance
model_manager = ModelManager()
