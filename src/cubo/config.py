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
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: config.json is malformed ({e}). Using default configuration.")
                return self._get_default_config()
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
            "supported_extensions": [".txt", ".docx", ".pdf", ".md"],
            "max_file_size_mb": 10,
            "rate_limit_seconds": 1,
            "log_level": "INFO",
            "log_file": "./logs/rag_log.txt",
            "vector_db_path": "./chroma_db",
            "faiss_index_dir": "./faiss_index",
            "vector_store_backend": "faiss",
            "vector_store_path": "./faiss_index",
            "chroma_db_path": "./chroma_db",
            "auto_merging_chunk_sizes": [2048, 512, 128],
            "auto_merging_collection_name": "cubo_auto_merging",
            "auto_merging_candidate_multiplier": 3,
            "auto_merging_parent_similarity_threshold": 0.1,
            "deep_output_dir": "./data/deep",
            "deep_csv_rows_per_chunk": 25,
            "deep_chunk_id_use_file_hash": True,
            "metadata_db_path": "./data/metadata.db",
            "ingestion": {
                "fast_pass": {
                    "output_dir": "data/fastpass",
                    "skip_heavy_models": True,
                    "auto_trigger_deep": False
                },
                "deep": {
                    "output_dir": "./data/deep",
                    "csv_rows_per_chunk": 25,
                    "use_file_hash_for_chunk_id": True,
                    "n_workers": 2
                },
                "chunking": {
                    "method": "sentence_window",
                    "use_sentence_window": True,
                    "window_size": 3,
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                }
            }
            ,
            "routing": {
                "enable": True,
                "factual_bm25_weight": 0.6,
                "conceptual_dense_weight": 0.8
            }
            ,
            "vector_index": {
                "hot_ratio": 0.2,
                "promote_threshold": 10,
                "nlist": 4096,
                "pq_m": 64
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking environment variables first."""
        # Support nested keys with dot notation (e.g., ingestion.fast_pass.output_dir)
        env_key = f"CUBO_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Traverse nested keys
        parts = key.split('.') if isinstance(key, str) and '.' in key else [key]
        cur = self._config
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def update(self, settings: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        # Support nested dicts merging
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(self._config, settings)

    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=4)
        except IOError as e:
            print(f"Error saving configuration to {self.config_path}: {e}")

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


# Global config instance
config = Config()
