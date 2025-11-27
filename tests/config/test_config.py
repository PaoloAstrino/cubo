import json
import os
import tempfile

from src.cubo.config import Config


def test_config_load_defaults():
    """Test loading default config when no file exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        config = Config(config_path)
        assert config.get("model_path") == "./models/embeddinggemma-300m"
        assert config.get("llm_model") == "llama3.2:latest"


def test_config_load_from_file():
    """Test loading config from existing file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        test_data = {"model_path": "/custom/path", "top_k": 5}
        with open(config_path, "w") as f:
            json.dump(test_data, f)
        config = Config(config_path)
        assert config.get("model_path") == "/custom/path"
        assert config.get("top_k") == 5


def test_config_set_and_save():
    """Test setting values and saving config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        config = Config(config_path)
        config.set("new_key", "new_value")
        config.save()
        # Reload and check
        config2 = Config(config_path)
        assert config2.get("new_key") == "new_value"


def test_config_all():
    """Test getting all config values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        config = Config(config_path)
        all_config = config.all
        assert isinstance(all_config, dict)
        assert "model_path" in all_config
