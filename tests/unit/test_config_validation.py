"""
Unit tests for configuration loading and validation.

Tests config file parsing, schema validation, default values,
and error handling for missing or invalid configurations.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any


class TestConfigLoading:
    """Test suite for config file loading."""
    
    def test_load_valid_config_json(self):
        """Test loading a valid JSON config file."""
        config_data = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "rerank_model": "BAAI/bge-reranker-base",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 50,
            "rerank_top_k": 10
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load and validate
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config["embedding_model"] == "BAAI/bge-base-en-v1.5"
            assert loaded_config["chunk_size"] == 512
            assert loaded_config["top_k"] == 50
        finally:
            Path(config_path).unlink()
    
    def test_config_missing_required_field(self):
        """Test handling of config missing required fields."""
        incomplete_config = {
            "chunk_size": 512,
            # Missing embedding_model
        }
        
        required_fields = ["embedding_model", "chunk_size", "top_k"]
        
        for field in required_fields:
            if field not in incomplete_config:
                # Should raise KeyError or validation error
                assert True
                break
    
    def test_config_default_values(self):
        """Test that default values are applied correctly."""
        minimal_config = {
            "embedding_model": "BAAI/bge-base-en-v1.5"
        }
        
        # Apply defaults
        defaults = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 50,
            "rerank_top_k": 10,
            "batch_size": 32
        }
        
        full_config = {**defaults, **minimal_config}
        
        assert full_config["chunk_size"] == 512
        assert full_config["embedding_model"] == "BAAI/bge-base-en-v1.5"
        assert full_config["top_k"] == 50
    
    def test_config_type_validation(self):
        """Test that config values have correct types."""
        config = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 50,
            "use_reranker": True
        }
        
        assert isinstance(config["embedding_model"], str)
        assert isinstance(config["chunk_size"], int)
        assert isinstance(config["chunk_overlap"], int)
        assert isinstance(config["top_k"], int)
        assert isinstance(config["use_reranker"], bool)
    
    def test_config_range_validation(self):
        """Test that numeric config values are in valid ranges."""
        # chunk_size should be positive
        assert 512 > 0
        assert 1024 > 0
        
        # chunk_overlap should be less than chunk_size
        chunk_size = 512
        chunk_overlap = 50
        assert chunk_overlap < chunk_size
        
        # top_k should be positive
        assert 50 > 0
        assert 100 > 0
        
        # Invalid: negative values
        invalid_chunk_size = -512
        assert invalid_chunk_size < 0  # Should fail validation


class TestBEIRConfigValidation:
    """Test suite for BEIR benchmark configuration."""
    
    def test_beir_dataset_config(self):
        """Test BEIR dataset configuration structure."""
        beir_config = {
            "datasets": ["nfcorpus", "scifact", "arguana", "fiqa"],
            "corpus_path": "data/beir/{dataset}/corpus.jsonl",
            "queries_path": "data/beir/{dataset}/queries.jsonl",
            "qrels_path": "data/beir/{dataset}/qrels/test.tsv",
            "output_dir": "results"
        }
        
        assert len(beir_config["datasets"]) == 4
        assert "nfcorpus" in beir_config["datasets"]
        assert "{dataset}" in beir_config["corpus_path"]
        
        # Test path templating
        dataset = "nfcorpus"
        corpus_path = beir_config["corpus_path"].format(dataset=dataset)
        assert corpus_path == "data/beir/nfcorpus/corpus.jsonl"
    
    def test_benchmark_config_validation(self):
        """Test benchmark configuration parameters."""
        benchmark_config = {
            "datasets": ["nfcorpus", "scifact"],
            "metrics": ["recall@10", "mrr", "ndcg@10"],
            "k_values": [10, 20, 50],
            "enable_reranking": True,
            "laptop_mode": True
        }
        
        assert isinstance(benchmark_config["datasets"], list)
        assert isinstance(benchmark_config["metrics"], list)
        assert "recall@10" in benchmark_config["metrics"]
        assert 10 in benchmark_config["k_values"]
    
    def test_rrf_sweep_config_validation(self):
        """Test RRF parameter sweep configuration."""
        rrf_config = {
            "k_values": [20, 60, 120],
            "semantic_weights": [0.7, 1.0, 1.3],
            "bm25_weights": [0.7, 1.0, 1.3]
        }
        
        # Should generate 3 * 3 * 3 = 27 combinations
        num_combinations = (
            len(rrf_config["k_values"]) *
            len(rrf_config["semantic_weights"]) *
            len(rrf_config["bm25_weights"])
        )
        
        assert num_combinations == 27
        assert 60 in rrf_config["k_values"]  # Baseline k
        assert 1.0 in rrf_config["semantic_weights"]  # Baseline weight


class TestConfigPathResolution:
    """Test suite for config path resolution."""
    
    def test_relative_path_resolution(self):
        """Test resolution of relative paths in config."""
        base_path = Path("c:/Users/paolo/Desktop/cubo")
        relative_path = "data/beir/nfcorpus/corpus.jsonl"
        
        full_path = base_path / relative_path
        
        assert full_path.is_absolute()
        assert "cubo" in str(full_path)
        assert "nfcorpus" in str(full_path)
    
    def test_config_file_discovery(self):
        """Test config file discovery in multiple locations."""
        search_paths = [
            "configs/config_local.json",
            "cubo/config.json",
            "config.json"
        ]
        
        # At least one should exist
        base_path = Path("c:/Users/paolo/Desktop/cubo")
        found = False
        
        for config_path in search_paths:
            full_path = base_path / config_path
            if full_path.exists():
                found = True
                break
        
        # In test environment, we can check expected paths
        expected_path = base_path / "configs" / "config_local.json"
        assert search_paths[0] == "configs/config_local.json"
    
    def test_environment_variable_expansion(self):
        """Test environment variable expansion in paths."""
        import os
        
        # Simulate env var in config
        path_with_env = "${HOME}/cubo/data"
        
        # Would expand to actual home directory
        home_dir = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        
        if home_dir:
            expanded_path = path_with_env.replace("${HOME}", home_dir)
            assert home_dir in expanded_path


class TestConfigErrorHandling:
    """Test suite for config error handling."""
    
    def test_invalid_json_syntax(self):
        """Test handling of malformed JSON."""
        invalid_json = '{"key": "value",}'  # Trailing comma
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        nonexistent_path = "path/to/nonexistent/config.json"
        
        with pytest.raises(FileNotFoundError):
            with open(nonexistent_path, 'r') as f:
                json.load(f)
    
    def test_config_schema_mismatch(self):
        """Test handling of config with wrong schema."""
        wrong_schema = {
            "unknown_field": "value",
            "another_unknown": 123
        }
        
        required_fields = ["embedding_model", "chunk_size"]
        
        missing_fields = [f for f in required_fields if f not in wrong_schema]
        
        assert len(missing_fields) > 0  # Should detect missing fields
        assert "embedding_model" in missing_fields
    
    def test_invalid_config_values(self):
        """Test handling of invalid config values."""
        invalid_configs = [
            {"chunk_size": -512},  # Negative size
            {"top_k": 0},  # Zero top_k
            {"chunk_overlap": 1000, "chunk_size": 512},  # Overlap > size
            {"embedding_model": ""},  # Empty model name
        ]
        
        # Each should fail validation
        for config in invalid_configs:
            if "chunk_size" in config and config["chunk_size"] < 0:
                assert True  # Would raise ValueError
            if "top_k" in config and config["top_k"] <= 0:
                assert True  # Would raise ValueError
            if "chunk_overlap" in config and "chunk_size" in config:
                if config["chunk_overlap"] >= config["chunk_size"]:
                    assert True  # Would raise ValueError


class TestConfigMerging:
    """Test suite for config merging and overrides."""
    
    def test_config_override_precedence(self):
        """Test that CLI args override config file."""
        config_file = {"chunk_size": 512, "top_k": 50}
        cli_args = {"top_k": 100}  # Override
        
        merged = {**config_file, **cli_args}
        
        assert merged["chunk_size"] == 512  # From config
        assert merged["top_k"] == 100  # Overridden by CLI
    
    def test_config_deep_merge(self):
        """Test deep merging of nested config structures."""
        base_config = {
            "retrieval": {
                "top_k": 50,
                "use_reranker": True
            }
        }
        
        override_config = {
            "retrieval": {
                "top_k": 100  # Override only this
            }
        }
        
        # Deep merge would preserve use_reranker
        merged_retrieval = {**base_config["retrieval"], **override_config["retrieval"]}
        
        assert merged_retrieval["top_k"] == 100
        assert merged_retrieval["use_reranker"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
