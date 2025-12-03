import json
import os
from typing import Any, Dict, Tuple


def _detect_system_resources() -> Tuple[float, int]:
    """Detect system RAM (GB) and CPU core count.

    Returns:
        Tuple of (ram_gb, cpu_count). Falls back to conservative defaults.
    """
    ram_gb = 16.0  # Conservative default
    cpu_count = 4  # Conservative default

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
    except ImportError:
        # psutil not available, try os-level detection
        try:
            cpu_count = os.cpu_count() or 4
        except Exception:
            pass
        # Try to read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024**2)
                        break
        except Exception:
            pass

    return ram_gb, cpu_count


def _should_enable_laptop_mode() -> bool:
    """Determine if laptop mode should be auto-enabled based on system resources.

    Laptop mode is enabled by default if:
    - RAM <= 16GB OR CPU cores <= 6
    - AND CUBO_LAPTOP_MODE env var is not explicitly set to '0' or 'false'

    Returns:
        True if laptop mode should be enabled.
    """
    # Check explicit opt-out
    env_val = os.environ.get("CUBO_LAPTOP_MODE", "").lower()
    if env_val in ("0", "false", "no", "off"):
        return False
    if env_val in ("1", "true", "yes", "on"):
        return True

    # Auto-detect based on resources
    ram_gb, cpu_count = _detect_system_resources()
    return ram_gb <= 16 or cpu_count <= 6


class Config:
    """Configuration management for CUBO."""

    def __init__(self, config_path: str = None):
        # Allow override using environment variable before defaulting to packaged config
        env_config_path = os.getenv("CUBO_CONFIG_PATH")
        if config_path is None:
            if env_config_path:
                config_path = env_config_path
            else:
                config_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "config.json"
                )
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file or use defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
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
            "faiss_index_dir": "./data/faiss",
            "faiss_index_root": None,
            "vector_store_path": "./data/faiss",
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
                    "auto_trigger_deep": False,
                },
                "deep": {
                    "output_dir": "./data/deep",
                    "csv_rows_per_chunk": 25,
                    "use_file_hash_for_chunk_id": True,
                    "n_workers": 2,
                    "enrich_enabled": False,  # Disabled by default for laptop compatibility
                    "chunk_batch_size": 100,  # Streaming save batch size
                    "auto_generate_scaffolds": False,
                },
                "chunking": {
                    "method": "sentence_window",
                    "use_sentence_window": True,
                    "window_size": 3,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                },
            },
            "document_cache_size": 1000,  # LRU cache for document content
            "routing": {"enable": True, "factual_bm25_weight": 0.6, "conceptual_dense_weight": 0.8},
            "vector_index": {
                "hot_ratio": 0.1,
                "promote_threshold": 50,
                "nlist": 1024,
                "pq_m": 64,
                "hnsw_m": 12,
                "hnsw_ef": 64,
                "normalize": True,
            },
            "deduplication": {
                "enabled": True,
                "method": "hybrid",
                "run_on": "scaffold",
                "representative_metric": "summary_score",
                "similarity_threshold": 0.92,
                "map_path": "output/dedup_clusters.json",
                "prefilter": {"use_minhash": True, "num_perm": 128, "minhash_threshold": 0.8},
                "ann": {"backend": "faiss", "k": 50},
                "clustering": {
                    "algorithm": "hdbscan",
                    "min_cluster_size": 2,
                    "min_samples": 1,
                    "umap_dims": 32,
                },
            },
            "llm": {
                "provider": "ollama",
                "model_name": "llama3",
                "system_prompt": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate.",
                "enable_streaming": False,
            },
            "scaffold": {
                "use_semantic_clustering": False,  # Enable for better scaffold quality
                "clustering_method": "kmeans",  # or 'hdbscan'
                "scaffold_size": 5,
                "min_cluster_size": 3,
            },
        }

    @staticmethod
    def get_laptop_mode_config() -> Dict[str, Any]:
        """Optimized config for resource-constrained laptops.

        This config reduces memory and CPU usage while maintaining reasonable
        retrieval quality. Key optimizations:

        **Model Management** (NEW - saves 300-800MB RAM):
        - Lazy model loading: Load on first use, unload after 5 min idle
        - Model idle timeout: 300s (5 minutes)

        **Vector Store** (NEW - saves 80-90% RAM for embeddings):
        - Memory-mapped embeddings: Store on disk, load on-demand
        - Smaller document cache: 500 docs instead of 1000

        **Retrieval** (saves RAM and computation):
        - Disable summary prefilter (saves 2x embeddings)
        - Disable scaffold compression (saves clustering overhead)
        - Disable cross-encoder reranking
        - Enable semantic cache for query results

        **Ingestion** (biggest CPU/GPU saver):
        - Disable LLM chunk enrichment
        - Single worker, small batches
        - No auto-scaffold generation

        **Total RAM Savings**: ~1-2GB compared to default mode
        """
        return {
            "laptop_mode": True,
            # Model management - NEW lazy loading
            "model_lazy_loading": True,  # Enable lazy model loading
            "model_idle_timeout": 300,  # Unload model after 5 min idle (300s)
            # Document cache
            "document_cache_size": 500,  # Smaller cache for low-RAM systems
            # Ingestion optimizations
            "ingestion": {
                "deep": {
                    "enrich_enabled": False,  # Critical: disables LLM enrichment
                    "n_workers": 1,
                    "batch_size": 5,
                    "chunk_batch_size": 50,  # Smaller batches for streaming saves
                    "throttle_delay_ms": 500,
                    "auto_generate_scaffolds": False,  # Disable scaffolds
                }
            },
            # Retrieval optimizations
            "retrieval": {
                "reranker_model": None,  # Disable cross-encoder reranking
                "semantic_cache": {"enabled": True, "threshold": 0.92, "max_entries": 500},
                # Disable tiered retrieval features (saves RAM)
                "use_summary_prefilter": False,  # Saves 2x embeddings
                "use_scaffold_compression": False,  # Saves clustering overhead
            },
            # Vector store - NEW memory-mapped mode
            "vector_store": {
                "embedding_storage": "mmap",  # NEW: Memory-mapped embeddings
                # Backwards-compatible key name expected by tests & other code
                "persist_embeddings": "npy_sharded",
                # Use float16 to save memory on laptop mode
                "embedding_dtype": "float16",
                "embedding_cache_size": 512,
            },
            # FAISS index optimizations
            "vector_index": {
                "hot_ratio": 0.1,  # Keep only 10% in hot index
                "promote_threshold": 100,  # Higher threshold for promotions
                "nlist": 512,  # Smaller clustering
                "pq_m": 32,  # Product quantization
                "hnsw_m": 8,  # Smaller graph for laptop
                "hnsw_ef": 32,  # Faster search
                "normalize": True,
            },
            # Deduplication limits
            "deduplication": {"max_candidates": 200},
        }

    def apply_laptop_mode(self, force: bool = False) -> bool:
        """Apply laptop mode configuration if appropriate.

        Args:
            force: If True, apply laptop mode regardless of system detection.

        Returns:
            True if laptop mode was applied, False otherwise.
        """
        if force or _should_enable_laptop_mode():
            self.update(Config.get_laptop_mode_config())
            return True
        return False

    def apply_default_mode(self, force: bool = False) -> bool:
        """Revert to the default (non-laptop) configuration.

        Args:
            force: If True, force reverting to default mode.

        Returns:
            True if default mode is applied.
        """
        if force or not _should_enable_laptop_mode():
            # Reload base defaults from the packaged config (or from path)
            self._config = self._load_config()
            # Ensure laptop_mode flag is cleared
            self.set("laptop_mode", False)
            return True
        return False

    def is_laptop_mode(self) -> bool:
        """Check if laptop mode is currently enabled."""
        return bool(self.get("laptop_mode", False))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value, checking environment variables first."""
        # Support nested keys with dot notation (e.g., ingestion.fast_pass.output_dir)
        env_key = f"CUBO_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Traverse nested keys
        parts = key.split(".") if isinstance(key, str) and "." in key else [key]
        cur = self._config
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, key: str, value: Any) -> None:
        """Set configuration value. Supports nested keys using dot notation.

        Example: set('logging.log_file', './logs/app.jsonl') will create/update nested dicts.
        """
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            cur = self._config
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value
        else:
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
            with open(self.config_path, "w") as f:
                json.dump(self._config, f, indent=4)
        except OSError as e:
            print(f"Error saving configuration to {self.config_path}: {e}")

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


# Global config instance
config = Config()

# Auto-enable laptop mode based on system resources (opt-out via CUBO_LAPTOP_MODE=0)
_laptop_mode_applied = config.apply_laptop_mode()
if _laptop_mode_applied:
    import logging as _logging

    _logging.getLogger("cubo.config").info(
        "Laptop mode auto-enabled based on system resources. " "Set CUBO_LAPTOP_MODE=0 to disable."
    )


# Logging configuration
import logging
import logging.config
from pathlib import Path
from typing import Optional

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/cubo.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {"cubo": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}},
    "root": {"level": "INFO", "handlers": ["console"]},
}


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to 'logs' in current directory.
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging_config = LOGGING_CONFIG.copy()
    logging_config["handlers"]["file"]["filename"] = str(log_dir / "cubo.log")

    level = level.upper()
    logging_config["handlers"]["console"]["level"] = level
    logging_config["loggers"]["cubo"]["level"] = level
    logging_config["root"]["level"] = level

    logging.config.dictConfig(logging_config)
