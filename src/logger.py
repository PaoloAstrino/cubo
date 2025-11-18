from src.cubo.utils.logger import *

__all__ = [name for name in dir() if not name.startswith('_')]
    """Logger class for CUBO."""

    def __init__(self):
        self.logger = None
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        # Ensure log directory exists
        log_dir = os.path.dirname(config.get("log_file"))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Prevent adding duplicate handlers if already configured
        if self.logger and self.logger.handlers:
            return

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.get("log_level", "INFO")),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.get("log_file")),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def get_logger(self):
        """Get the logger instance."""
        return self.logger


# Global logger instance
logger_instance = Logger()
logger = logger_instance.get_logger()
