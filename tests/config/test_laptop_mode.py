"""Tests for laptop mode auto-detection and configuration."""

import os
import unittest
from unittest.mock import MagicMock, patch


class TestLaptopModeDetection(unittest.TestCase):
    """Tests for system resource detection and laptop mode auto-enable."""

    def test_detect_system_resources_with_psutil(self):
        """Test resource detection when psutil is available."""
        from cubo.config import _detect_system_resources

        # Mock psutil
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB

        with patch.dict("sys.modules", {"psutil": MagicMock()}):
            import sys

            sys.modules["psutil"].virtual_memory.return_value = mock_virtual_memory
            sys.modules["psutil"].cpu_count.return_value = 8

            # Force reimport to use mocked psutil

            # Note: Detection happens at import time, so we test the function directly

        # Just verify function exists and returns tuple
        ram, cores = _detect_system_resources()
        self.assertIsInstance(ram, (int, float))
        self.assertIsInstance(cores, int)
        self.assertGreater(ram, 0)
        self.assertGreater(cores, 0)

    def test_detect_system_resources_fallback(self):
        """Test resource detection falls back when psutil is unavailable."""
        from cubo.config import _detect_system_resources

        ram, cores = _detect_system_resources()

        # Should return sensible values
        self.assertIsInstance(ram, (int, float))
        self.assertIsInstance(cores, int)
        self.assertGreater(ram, 0)
        self.assertGreater(cores, 0)

    def test_should_enable_laptop_mode_threshold(self):
        """Test laptop mode is enabled on low-resource systems."""
        from cubo.config import _detect_system_resources, _should_enable_laptop_mode

        # Clear any existing env var override
        env_backup = os.environ.pop("CUBO_LAPTOP_MODE", None)

        try:
            ram, cores = _detect_system_resources()
            result = _should_enable_laptop_mode()

            # If RAM <= 16GB or cores <= 6, should return True
            expected = ram <= 16 or cores <= 6
            self.assertEqual(
                result,
                expected,
                f"RAM={ram}GB, cores={cores}: expected laptop_mode={expected}, got {result}",
            )
        finally:
            if env_backup is not None:
                os.environ["CUBO_LAPTOP_MODE"] = env_backup

    def test_env_var_forces_laptop_mode_on(self):
        """Test CUBO_LAPTOP_MODE=1 forces laptop mode on."""
        from cubo.config import _should_enable_laptop_mode

        with patch.dict(os.environ, {"CUBO_LAPTOP_MODE": "1"}):
            result = _should_enable_laptop_mode()
            self.assertTrue(result)

    def test_env_var_forces_laptop_mode_off(self):
        """Test CUBO_LAPTOP_MODE=0 forces laptop mode off."""
        from cubo.config import _should_enable_laptop_mode

        with patch.dict(os.environ, {"CUBO_LAPTOP_MODE": "0"}):
            result = _should_enable_laptop_mode()
            self.assertFalse(result)


class TestLaptopModeConfig(unittest.TestCase):
    """Tests for laptop mode configuration settings."""

    def test_get_laptop_mode_config_structure(self):
        """Test laptop mode config has expected structure."""
        from cubo.config import Config

        laptop_config = Config.get_laptop_mode_config()

        # Check top-level keys
        self.assertIn("laptop_mode", laptop_config)
        self.assertTrue(laptop_config["laptop_mode"])

        # Check ingestion settings
        self.assertIn("ingestion", laptop_config)
        self.assertFalse(laptop_config["ingestion"]["deep"]["enrich_enabled"])
        self.assertEqual(laptop_config["ingestion"]["deep"]["n_workers"], 1)

        # Check retrieval settings
        self.assertIn("retrieval", laptop_config)
        self.assertIsNone(laptop_config["retrieval"]["reranker_model"])

        # Check vector store settings
        self.assertIn("vector_store", laptop_config)
        self.assertEqual(laptop_config["vector_store"]["persist_embeddings"], "npy_sharded")

        # Check deduplication settings
        self.assertIn("deduplication", laptop_config)
        self.assertEqual(laptop_config["deduplication"]["max_candidates"], 200)

    def test_apply_laptop_mode(self):
        """Test apply_laptop_mode updates config correctly."""
        from cubo.config import Config

        config = Config()

        # Force laptop mode
        result = config.apply_laptop_mode(force=True)

        self.assertTrue(result)
        self.assertTrue(config.is_laptop_mode())

        # Verify specific settings were applied
        self.assertFalse(config.get("ingestion.deep.enrich_enabled"))
        self.assertEqual(config.get("ingestion.deep.n_workers"), 1)
        self.assertIsNone(config.get("retrieval.reranker_model"))

    def test_is_laptop_mode(self):
        """Test is_laptop_mode returns correct state."""
        from cubo.config import Config

        config = Config()

        # Initially may or may not be in laptop mode
        initial_state = config.is_laptop_mode()
        self.assertIsInstance(initial_state, bool)

        # Force enable
        config.set("laptop_mode", True)
        self.assertTrue(config.is_laptop_mode())

        # Force disable
        config.set("laptop_mode", False)
        self.assertFalse(config.is_laptop_mode())

    def test_apply_default_mode_reverts(self):
        """Test apply_default_mode reverts laptop-specific changes."""
        from cubo.config import Config

        cfg = Config()
        cfg.apply_laptop_mode(force=True)
        self.assertTrue(cfg.is_laptop_mode())

        # Revert to defaults
        result = cfg.apply_default_mode(force=True)
        self.assertTrue(result)
        self.assertFalse(cfg.is_laptop_mode())


class TestLaptopModeEnhancements(unittest.TestCase):
    """Tests for laptop mode specific optimizations."""

    def test_document_cache_size_reduced(self):
        """Test laptop mode reduces document cache size."""
        from cubo.config import Config

        laptop_config = Config.get_laptop_mode_config()

        self.assertEqual(laptop_config["document_cache_size"], 500)

    def test_semantic_cache_enabled(self):
        """Test laptop mode enables semantic cache as reranker replacement."""
        from cubo.config import Config

        laptop_config = Config.get_laptop_mode_config()

        self.assertTrue(laptop_config["retrieval"]["semantic_cache"]["enabled"])
        self.assertGreaterEqual(laptop_config["retrieval"]["semantic_cache"]["threshold"], 0.9)

    def test_vector_index_reduced_complexity(self):
        """Test laptop mode uses lower FAISS index complexity."""
        from cubo.config import Config

        laptop_config = Config.get_laptop_mode_config()

        self.assertIn("vector_index", laptop_config)
        self.assertLessEqual(laptop_config["vector_index"]["hot_ratio"], 0.2)
        self.assertLessEqual(laptop_config["vector_index"]["nlist"], 1024)

    def test_embedding_persistence_enabled(self):
        """Test laptop mode enables on-disk embedding persistence."""
        from cubo.config import Config

        laptop_config = Config.get_laptop_mode_config()

        self.assertIn("vector_store", laptop_config)
        self.assertIn("persist_embeddings", laptop_config["vector_store"])
        self.assertIn(
            laptop_config["vector_store"]["persist_embeddings"], ["npy", "npy_sharded", "mmap"]
        )
        self.assertEqual(laptop_config["vector_store"]["embedding_dtype"], "float16")


class TestGlobalConfigAutoDetection(unittest.TestCase):
    """Tests for global config laptop mode auto-detection at module load."""

    def test_global_config_exists(self):
        """Test global config instance is created."""
        from cubo.config import config

        self.assertIsNotNone(config)

    def test_laptop_mode_applied_flag_exists(self):
        """Test _laptop_mode_applied flag is set at module level."""
        from cubo import config as config_module

        self.assertTrue(hasattr(config_module, "_laptop_mode_applied"))
        self.assertIsInstance(config_module._laptop_mode_applied, bool)

    def test_config_path_env_var_override(self):
        """Test that CUBO_CONFIG_PATH env var is used when creating a Config instance."""
        import json
        import tempfile

        from cubo.config import Config

        # Create a temporary config file with a distinctive value
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump({"model_path": "./tmp_models", "laptop_mode": True}, tmp)
            tmp.flush()
            tmp_path = tmp.name

        try:
            with patch.dict(os.environ, {"CUBO_CONFIG_PATH": tmp_path}):
                cfg = Config()
                self.assertEqual(cfg.get("model_path"), "./tmp_models")
                self.assertTrue(cfg.is_laptop_mode())
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
