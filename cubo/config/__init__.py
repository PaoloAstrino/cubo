import json
import os
from pathlib import Path
from typing import Tuple

from pydantic import BaseModel as _PydanticBaseModel

from cubo.config.settings import Settings, settings


class ConfigAdapter:
    """Adapter to preserve old config.get/set API while backed by Pydantic settings."""

    def __init__(self, settings_obj: Settings):
        self._settings = settings_obj
        self._overrides = {}  # store keys not represented in Settings

    def get(self, key: str, default=None):
        # Overrides take precedence for non-modeled keys
        if key in self._overrides:
            return self._overrides.get(key, default)

        current = self._settings
        for part in key.split("."):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return self._overrides.get(key, default)
        # Normalize Pydantic models to plain dicts for backward compatibility where callers
        # expect `.get()` on nested config sections.
        if isinstance(current, _PydanticBaseModel):
            try:
                return current.model_dump()
            except Exception:
                # Fallback: convert via json roundtrip
                return json.loads(current.model_dump_json())
        return current

    def set(self, key: str, value):
        parts = key.split(".")
        target = self._settings
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            elif isinstance(target, dict):
                target = target.get(part, {})
            else:
                self._overrides[key] = value
                return

        leaf = parts[-1]
        if hasattr(target, leaf):
            existing = getattr(target, leaf)
            # If existing is a Pydantic model and new value is a dict, apply nested keys
            if isinstance(existing, _PydanticBaseModel) and isinstance(value, dict):
                for k, v in value.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                    elif isinstance(existing, dict):
                        existing[k] = v
                    else:
                        # fallback to overrides if nested attr doesn't exist
                        self._overrides[f"{key}.{k}"] = v
            else:
                setattr(target, leaf, value)
        elif isinstance(target, dict):
            target[leaf] = value
        else:
            self._overrides[key] = value

    @property
    def all(self):
        combined = json.loads(self._settings.model_dump_json())
        combined.update(self._overrides)
        return combined


class Config(ConfigAdapter):
    """Compatibility wrapper that mirrors the legacy Config class backed by Settings."""

    def __init__(self, config_path: str | None = None):
        # Allow env var override for tests and user configuration
        env_path = os.environ.get("CUBO_CONFIG_PATH")
        if not config_path and env_path:
            config_path = env_path
        self._config_path = Path(config_path) if config_path else None
        super().__init__(settings)
        self._load_defaults()
        self._load_file()

    def update(self, changes: dict):
        """Update config in bulk from a dict of dotted keys or nested dict.

        Accepts either {'a.b': v} or nested {'a': {'b': v}}.
        """

        def _recurse(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        _recurse(new_prefix, v)
                    else:
                        self.set(new_prefix, v)
            else:
                self.set(prefix, obj)

        _recurse("", changes)

    @property
    def _config(self) -> dict:
        # Backwards-compatibility property used directly by tests
        return self.all

    @_config.setter
    def _config(self, value: dict):
        # Accept a nested dict and apply values to config via set()
        if not isinstance(value, dict):
            return
        for k, v in value.items():
            if isinstance(v, dict):
                self.update({k: v})
            else:
                self.set(k, v)

    def _load_defaults(self):
        # Legacy defaults expected by tests and older callers
        defaults = {
            "model_path": "./models/embeddinggemma-300m",
            "data_folder": "./data",
            "supported_extensions": [".txt", ".docx", ".pdf", ".md"],
            "log_file": "./logs/cubo_log.jsonl",
            "llm_model": "llama3.2:latest",
            "llm": {"n_ctx": 0},  # 0 = auto-detect
            "vector_store": {"embedding_dtype": "float16"},
        }
        for key, value in defaults.items():
            self.set(key, value)

    def _load_file(self):
        if not self._config_path or not self._config_path.exists():
            return
        try:
            data = json.loads(self._config_path.read_text())
            if isinstance(data, dict):
                for key, value in data.items():
                    self.set(key, value)
        except Exception:
            # Ignore malformed files to preserve runtime behavior
            return

    def save(self):
        if not self._config_path:
            return
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.all, indent=2)
        self._config_path.write_text(payload)

    @staticmethod
    def get_laptop_mode_config() -> dict:
        """Return defaults for the laptop-mode configuration as a plain dictionary.

        Tests expect a well-formed object with specific keys/values.
        """
        return {
            "laptop_mode": True,
            "document_cache_size": 500,
            "ingestion": {
                "deep": {
                    "enrich_enabled": False,
                    "n_workers": 1,
                }
            },
            "retrieval": {
                "reranker_model": None,
                "semantic_cache": {"enabled": True, "threshold": 0.9},
            },
            "vector_store": {
                "persist_embeddings": "npy_sharded",
                "embedding_storage": "mmap",
                "embedding_dtype": "float16",
            },
            "deduplication": {"max_candidates": 200},
            "vector_index": {"hot_ratio": 0.1, "nlist": 1024},
            "model_lazy_loading": True,
        }

    def is_laptop_mode(self) -> bool:
        return bool(self.get("laptop_mode", False))

    def apply_laptop_mode(self, force: bool = False) -> bool:
        """Apply laptop-mode configuration to this instance.

        Returns True if the config changed, False otherwise.
        """
        if self.is_laptop_mode() and not force:
            return False

        laptop_config = self.get_laptop_mode_config()

        # Dynamic worker count based on physical cores
        try:
            from cubo.utils.hardware import detect_hardware

            hw = detect_hardware()
            # Leave 1 core for system/UI
            n_workers = max(1, hw.physical_cores - 1)
            laptop_config["ingestion"]["deep"]["n_workers"] = n_workers

            # Intelligent Reranker Activation:
            # If we have AVX-512 (huge boost for CPU transformers) or enough cores,
            # we can afford a lightweight reranker instead of disabling it completely.
            has_avx512 = any("avx512" in f for f in hw.cpu_flags)
            if has_avx512 or hw.physical_cores >= 6:
                laptop_config["retrieval"][
                    "reranker_model"
                ] = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
        except Exception:
            pass  # Fallback to default 1 and no reranker

        self.update(laptop_config)

        # Universal GPU Acceleration: Check for hardware even in laptop mode
        try:
            from cubo.utils.hardware import detect_hardware

            hw = detect_hardware()
            if hw.device in ("cuda", "mps"):
                # Enable GPU layers for LLM (offload all)
                self.set("llm.n_gpu_layers", hw.n_gpu_layers)
                # Set device for embeddings
                self.set("embeddings.device", hw.device)
        except Exception:
            # Fallback if hardware detection fails
            pass

        self.set("laptop_mode", True)
        return True

    def apply_default_mode(self, force: bool = False) -> bool:
        """Revert laptop-mode changes (set laptop_mode False and reset known keys).

        Returns True if changes were applied.
        """
        if not self.is_laptop_mode() and not force:
            return False
        # Reset laptop-mode related keys to sane defaults
        self.set("laptop_mode", False)
        # Re-apply explicit defaults that are commonly changed in laptop mode
        self.set("document_cache_size", 500)
        # For ingestion/retrieval we keep it honest by reverting to typical defaults
        self.set("ingestion.deep.enrich_enabled", True)
        self.set("ingestion.deep.n_workers", 4)
        self.set("retrieval.reranker_model", "default")
        self.set("vector_store.persist_embeddings", "npy")
        self.set("vector_store.embedding_dtype", "float16")
        self.set("deduplication.max_candidates", 500)
        self.set("vector_index.hot_ratio", 0.5)
        self.set("vector_index.nlist", 4096)
        return True


# Backwards compatibility aliases
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"
config = Config(
    _DEFAULT_CONFIG_PATH
)  # legacy imports expecting config instance with defaults loaded

__all__ = ["Settings", "settings", "Config", "config", "ConfigAdapter"]


def _detect_system_resources() -> Tuple[int, int]:
    """Detects total RAM in GB and CPU cores.

    Returns (ram_gb, cpu_count).
    """
    try:
        import psutil

        mem = psutil.virtual_memory().total
        cores = psutil.cpu_count(logical=True) or 1
        ram_gb = int(mem / (1024 * 1024 * 1024))
        return ram_gb, int(cores)
    except Exception:
        # Fallback: best-effort using os and platform
        try:
            import os

            cores = os.cpu_count() or 1
            # If we can't get memory, fall back to 8GB conservative default
            return 8, cores
        except Exception:
            return 8, 2


def _should_enable_laptop_mode() -> bool:
    """Return whether laptop-mode should be enabled based on environment or heuristics.

    CUBO_LAPTOP_MODE env var: '1'/'true' => True, '0'/'false' => False.
    Otherwise, enable if RAM <= 16GB or cores <= 6.
    """
    val = os.environ.get("CUBO_LAPTOP_MODE")
    if val is not None:
        val_low = str(val).lower()
        if val_low in ("1", "true", "yes"):
            return True
        if val_low in ("0", "false", "no"):
            return False
    ram, cores = _detect_system_resources()
    return ram <= 16 or cores <= 6


# Apply laptop-mode at import time if heuristic / env var indicate so
_laptop_mode_applied = False
try:
    if _should_enable_laptop_mode():
        config.apply_laptop_mode(force=True)
        _laptop_mode_applied = True
except Exception:
    # Keep import-time effects minimal; failure here shouldn't crash
    _laptop_mode_applied = False
