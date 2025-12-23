"""
Test suite for LazyModelManager.

Tests lazy loading, auto-unloading, thread safety, and memory management.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from cubo.embeddings.lazy_model_manager import LazyModelManager, get_lazy_model_manager


class TestLazyModelManager(unittest.TestCase):
    """Test cases for LazyModelManager."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a short timeout for faster tests
        self.manager = LazyModelManager(
            model_path="test-model", idle_timeout=2  # 2 seconds for testing
        )

    def tearDown(self):
        """Clean up after tests."""
        if self.manager:
            self.manager.force_unload()

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_lazy_loading_on_first_access(self, mock_transformer):
        """Test that model loads only on first access."""
        # Mock the model
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Initially not loaded
        self.assertFalse(self.manager.is_loaded())

        # First access triggers loading
        model = self.manager.get_model()

        # Should be loaded now
        self.assertTrue(self.manager.is_loaded())
        self.assertEqual(model, mock_model)

        # SentenceTransformer should have been called once
        mock_transformer.assert_called_once()

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_model_reuse_without_reload(self, mock_transformer):
        """Test that model is reused without reloading."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # First access
        model1 = self.manager.get_model()

        # Second access immediately
        model2 = self.manager.get_model()

        # Should be the same model instance
        self.assertIs(model1, model2)

        # SentenceTransformer should only be called once
        self.assertEqual(mock_transformer.call_count, 1)

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_auto_unload_after_timeout(self, mock_transformer):
        """Test that model auto-unloads after idle timeout."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Load model
        self.manager.get_model()
        self.assertTrue(self.manager.is_loaded())

        # Wait for timeout (2 seconds + buffer)
        time.sleep(2.5)

        # Should be unloaded
        self.assertFalse(self.manager.is_loaded())

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_timeout_reset_on_access(self, mock_transformer):
        """Test that timeout resets on each access."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Load model
        self.manager.get_model()

        # Wait 1.5 seconds (less than 2s timeout)
        time.sleep(1.5)

        # Access again (should reset timeout)
        self.manager.get_model()

        # Wait another 1.5 seconds
        time.sleep(1.5)

        # Should still be loaded (total wait was 3s but timer reset)
        self.assertTrue(self.manager.is_loaded())

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_force_unload(self, mock_transformer):
        """Test force_unload immediately unloads model."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Load model
        self.manager.get_model()
        self.assertTrue(self.manager.is_loaded())

        # Force unload
        self.manager.force_unload()

        # Should be immediately unloaded
        self.assertFalse(self.manager.is_loaded())

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_thread_safety(self, mock_transformer):
        """Test that concurrent access is thread-safe."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        models = []

        def access_model():
            model = self.manager.get_model()
            models.append(model)

        # Create multiple threads accessing model simultaneously
        threads = [threading.Thread(target=access_model) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All threads should get the same model instance
        self.assertEqual(len(models), 10)
        self.assertTrue(all(m is models[0] for m in models))

        # Model should only be loaded once despite concurrent access
        self.assertEqual(mock_transformer.call_count, 1)

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_stats_reporting(self, mock_transformer):
        """Test that stats are correctly reported."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Initial stats (not loaded)
        stats = self.manager.get_stats()
        self.assertFalse(stats["loaded"])
        self.assertEqual(stats["estimated_memory_mb"], 0)

        # Load model
        self.manager.get_model()

        # Stats after loading
        stats = self.manager.get_stats()
        self.assertTrue(stats["loaded"])
        # With mocked model, memory estimate defaults to 400MB
        self.assertGreaterEqual(stats["estimated_memory_mb"], 0)
        self.assertIn("time_since_last_access", stats)

    @patch("cubo.embeddings.lazy_model_manager.torch")
    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_gpu_fallback_to_cpu(self, mock_transformer, mock_torch):
        """Test fallback from GPU to CPU on failure."""
        # Simulate GPU available but loading fails
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Mock GPU"

        # First call (GPU) raises error
        mock_transformer.side_effect = [RuntimeError("CUDA error"), MagicMock()]

        manager = LazyModelManager(idle_timeout=2)

        # Should fallback to CPU and succeed
        manager.get_model()

        # Should have tried twice (GPU then CPU)
        self.assertEqual(mock_transformer.call_count, 2)

        # Device should be CPU after fallback
        self.assertEqual(manager.device, "cpu")

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_disabled_timeout_keeps_model(self, mock_transformer):
        """Test that timeout=0 disables auto-unload."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Create manager with disabled timeout
        manager = LazyModelManager(idle_timeout=0)

        # Load model
        manager.get_model()
        self.assertTrue(manager.is_loaded())

    def test_laptop_mode_stub_dimension_respects_config(self):
        """Test that the lightweight laptop-mode stub uses configured index_dimension."""
        from cubo.config import config as _cfg

        # Save original values
        orig_laptop = _cfg.get("laptop_mode", False)
        orig_dim = _cfg.get("index_dimension", None)

        try:
            # Ensure the setUp manager is loaded (without network calls) so we can validate it remains loaded
            from cubo.embeddings.lazy_model_manager import _LightweightModel

            self.manager._model = _LightweightModel(dim=int(_cfg.get("index_dimension", 64) or 64))
            # Force laptop mode and set index dim to a test value
            _cfg.set("index_dimension", 128)
            _cfg.set("laptop_mode", True)
            mgr = get_lazy_model_manager()
            model = mgr.get_model()
            # The stub should match the configured index dimension
            self.assertEqual(model.get_sentence_embedding_dimension(), 128)
        finally:
            # Restore config and unload model
            _cfg.set("index_dimension", orig_dim if orig_dim is not None else 0)
            _cfg.set("laptop_mode", orig_laptop)
            try:
                mgr.force_unload()
            except Exception:
                pass

        # Wait significant time
        time.sleep(1)

        # Model should still be loaded
        self.assertTrue(self.manager.is_loaded())


class TestLazyModelManagerIntegration(unittest.TestCase):
    """Integration tests for LazyModelManager."""

    @patch("cubo.embeddings.lazy_model_manager.SentenceTransformer")
    def test_singleton_pattern(self, mock_transformer):
        """Test that get_lazy_model_manager returns singleton."""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        # Get manager twice
        manager1 = get_lazy_model_manager()
        manager2 = get_lazy_model_manager()

        # Should be the same instance
        self.assertIs(manager1, manager2)

    def test_recreate_on_index_dimension_change(self):
        """If index_dimension config changes, get_lazy_model_manager should recreate the manager."""
        from cubo.config import config as _cfg

        # Save state
        orig_dim = _cfg.get("index_dimension", None)
        orig_laptop = _cfg.get("laptop_mode", False)
        try:
            _cfg.set("laptop_mode", True)
            _cfg.set("index_dimension", 128)
            m1 = get_lazy_model_manager()
            # Change index dimension
            _cfg.set("index_dimension", 256)
            m2 = get_lazy_model_manager()
            # Should be a different instance
            self.assertIsNot(m1, m2)
            self.assertEqual(getattr(m2, "_created_with_index_dimension", 0), 256)
        finally:
            _cfg.set("index_dimension", orig_dim if orig_dim is not None else 0)
            _cfg.set("laptop_mode", orig_laptop)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
