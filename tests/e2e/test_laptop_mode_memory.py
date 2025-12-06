"""
Laptop Mode Memory Constraints Test

Tests that laptop mode keeps RAM usage under limits:
1. Enable laptop mode configuration
2. Ingest large dataset (1000+ documents)
3. Monitor RAM usage with psutil
4. Verify lazy model loading/unloading
5. Assert RAM < 2GB during operation
"""

import psutil
import time
from pathlib import Path

import pytest
pytest.importorskip("torch")

from cubo.config import Config
from cubo.embeddings.lazy_model_manager import get_lazy_model_manager
from cubo.retrieval.retriever import DocumentRetriever


def get_process_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


@pytest.fixture
def laptop_mode_config():
    """Configuration with laptop mode enabled."""
    config = Config()
    config.apply_laptop_mode(force=True)
    return config


@pytest.fixture
def large_document_set():
    """Generate a large set of documents for testing."""
    # Create 1000 short documents to simulate real workload
    documents = []
    for i in range(1000):
        category = i % 10
        doc = {
            "id": f"doc_{i:04d}",
            "text": f"Document {i} in category {category}. " * 10,  # ~100 words each
            "metadata": {"category": category}
        }
        documents.append(doc)
    return documents


class TestLaptopModeMemoryConstraints:
    """Test that laptop mode enforces RAM limits."""
    
    def test_laptop_mode_ram_under_2gb(self, laptop_mode_config, large_document_set):
        """
        Test that laptop mode keeps total RAM < 2GB during heavy operation.
        
        This is the core "potato laptop" promise.
        """
        # Record baseline memory
        baseline_mb = get_process_memory_mb()
        print(f"\nBaseline memory: {baseline_mb:.1f} MB")
        
        # Initialize retriever in laptop mode
        model_manager = get_lazy_model_manager()
        model = model_manager.get_model()
        
        memory_after_model = get_process_memory_mb()
        model_size_mb = memory_after_model - baseline_mb
        print(f"Model loaded: {model_size_mb:.1f} MB")
        
        retriever = DocumentRetriever(
            model=model,
            use_sentence_window=True,
            top_k=5
        )
        
        # Ingest 1000 documents
        max_memory_mb = baseline_mb
        memory_samples = []
        
        for i, doc in enumerate(large_document_set):
            retriever.add_document(doc["text"], metadata=doc["metadata"])
            
            # Sample memory every 100 documents
            if i % 100 == 0:
                current_mb = get_process_memory_mb()
                memory_samples.append(current_mb)
                max_memory_mb = max(max_memory_mb, current_mb)
                print(f"  {i} docs: {current_mb:.1f} MB")
        
        final_memory_mb = get_process_memory_mb()
        print(f"Final memory: {final_memory_mb:.1f} MB")
        print(f"Peak memory: {max_memory_mb:.1f} MB")
        
        # ASSERTION: Peak memory should be < 2048 MB (2 GB)
        assert max_memory_mb < 2048, \
            f"Laptop mode exceeded 2GB limit: {max_memory_mb:.1f} MB"
        
        print(f"✓ Laptop mode RAM constraint satisfied: {max_memory_mb:.1f} MB < 2048 MB")
    
    def test_lazy_model_unloading(self, laptop_mode_config):
        """
        Test that lazy model manager unloads model after idle timeout.
        
        This is critical for reducing RAM when not actively querying.
        """
        model_manager = get_lazy_model_manager()
        
        # Load model
        model = model_manager.get_model()
        assert model is not None, "Model not loaded"
        
        memory_with_model = get_process_memory_mb()
        print(f"\nMemory with model loaded: {memory_with_model:.1f} MB")
        
        # Simulate idle time (in real scenario, this would be 300s)
        # For testing, we manually trigger unload
        model_manager._last_used = time.time() - 400  # Fake old timestamp
        
        # Trigger unload check
        if hasattr(model_manager, '_maybe_unload_model'):
            model_manager._maybe_unload_model()
        
        memory_after_unload = get_process_memory_mb()
        print(f"Memory after unload: {memory_after_unload:.1f} MB")
        
        # Memory should decrease (at least 100 MB for small models)
        memory_freed = memory_with_model - memory_after_unload
        assert memory_freed > 50, \
            f"Model unload didn't free significant memory: only {memory_freed:.1f} MB"
        
        print(f"✓ Lazy unload freed {memory_freed:.1f} MB")
    
    def test_memory_mapped_embeddings(self, laptop_mode_config, tmp_path):
        """
        Test that memory-mapped embeddings reduce RAM usage.
        
        With mmap, embeddings stay on disk and are loaded on-demand.
        """
        # This test verifies the config setting
        assert laptop_mode_config.get("vector_store.embedding_storage") == "mmap", \
            "Laptop mode should use memory-mapped embeddings"
        
        print("✓ Laptop mode configured for memory-mapped embeddings")
    
    def test_batch_size_reduction(self, laptop_mode_config):
        """
        Test that laptop mode uses smaller batch sizes to reduce memory spikes.
        """
        # Verify laptop mode config has smaller batches
        batch_size = laptop_mode_config.get("ingestion.deep.chunk_batch_size", 100)
        
        assert batch_size <= 50, \
            f"Laptop mode batch size too large: {batch_size} (should be ≤50)"
        
        print(f"✓ Laptop mode using small batches: {batch_size} chunks/batch")
    
    def test_memory_efficiency_vs_default_mode(self, large_document_set):
        """
        Compare memory usage: laptop mode vs default mode.
        
        Laptop mode should use significantly less RAM.
        """
        # Laptop mode
        config_laptop = Config()
        config_laptop.apply_laptop_mode(force=True)
        
        baseline = get_process_memory_mb()
        
        model = get_lazy_model_manager().get_model()
        retriever_laptop = DocumentRetriever(model=model)
        
        # Add subset of documents
        for doc in large_document_set[:100]:
            retriever_laptop.add_document(doc["text"])
        
        laptop_memory = get_process_memory_mb() - baseline
        
        # Note: Full comparison would require resetting process
        # This is a simplified check
        assert laptop_memory < 1024, \
            f"Laptop mode used too much RAM: {laptop_memory:.1f} MB"
        
        print(f"✓ Laptop mode memory usage: {laptop_memory:.1f} MB for 100 docs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
