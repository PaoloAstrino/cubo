import json
from pathlib import Path

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
		for part in key.split('.'):
			if hasattr(current, part):
				current = getattr(current, part)
			elif isinstance(current, dict) and part in current:
				current = current[part]
			else:
				return self._overrides.get(key, default)
		return current

	def set(self, key: str, value):
		parts = key.split('.')
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
		self._config_path = Path(config_path) if config_path else None
		super().__init__(settings)
		self._load_defaults()
		self._load_file()

	def _load_defaults(self):
		# Legacy defaults expected by tests and older callers
		defaults = {
			"model_path": "./models/embeddinggemma-300m",
			"llm_model": "llama3.2:latest",
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


# Backwards compatibility aliases
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
config = Config(_DEFAULT_CONFIG_PATH)  # legacy imports expecting config instance with defaults loaded

__all__ = ["Settings", "settings", "Config", "config", "ConfigAdapter"]