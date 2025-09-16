import json
import os
from typing import Dict, Any

class Config:
    """Configuration management for CUBO."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "model_path": "./models/embeddinggemma-300m",
            "llm_model": "llama3.2:latest",
            "top_k": 3,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "data_folder": "./data",
            "supported_extensions": [".txt", ".docx", ".pdf"],
            "max_file_size_mb": 10,
            "rate_limit_seconds": 1,
            "log_level": "INFO",
            "log_file": "./logs/rag_log.txt"
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking environment variables first."""
        # Check environment variable first (e.g., CUBO_MODEL_PATH)
        env_key = f"CUBO_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=4)

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()

# Global config instance
config = Config()
