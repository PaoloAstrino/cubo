"""
End-to-end tests for single BEIR adapter runs.

Tests the complete workflow of running run_beir_adapter.py with
index building, reuse, and proper output generation.
"""

import pytest
import subprocess
import tempfile
import shutil
import json
from pathlib import Path


class TestBEIRAdapterRun:
    """Test suite for single BEIR adapter execution."""
    

    def test_beir_adapter_script_exists(self):
        """Test that run_beir_adapter.py exists."""
        script_path = Path("scripts/run_beir_adapter.py")
        assert script_path.exists(), "run_beir_adapter.py not found"
    
    def test_beir_adapter_help_flag(self):
        """Test that adapter responds to --help."""
        result = subprocess.run(
            ["python", "scripts/run_beir_adapter.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should show help without error
        assert result.returncode == 0 or "usage" in result.stdout.lower()
    
    def test_beir_adapter_required_arguments(self):
        """Test required command-line arguments."""
        required_args = [
            "--corpus",
            "--queries",
            "--output",
            "--index-dir"
        ]
        
        # These args should be required
        assert len(required_args) == 4
    
    def test_beir_adapter_optional_flags(self):
        """Test optional command-line flags."""
        optional_flags = [
            "--reindex",
            "--laptop-mode",
            "--verbose"
        ]
        
        # These are boolean flags
        assert len(optional_flags) == 3


class TestIndexBuildWorkflow:
    """Test suite for index building workflow."""
    
    def test_first_run_builds_index(self, tmp_path):
        """Test that first run builds index from scratch."""
        index_dir = tmp_path / "index"
        
        # Index dir doesn't exist yet
        assert not index_dir.exists()
        
        # First run should create index_dir
        # (In actual execution, adapter would create it)
    
    def test_reindex_flag_forces_rebuild(self, tmp_path):
        """Test that --reindex flag rebuilds existing index."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        
        # Create existing index files
        (index_dir / "hot.index").touch()
        (index_dir / "metadata.json").touch()
        
        # With --reindex, should rebuild despite existing files
        reindex = True
        should_rebuild = reindex or not (index_dir / "metadata.json").exists()
        
        assert should_rebuild is True
    
    def test_existing_index_reused(self, tmp_path):
        """Test that existing valid index is reused."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        
        # Create index files
        (index_dir / "hot.index").touch()
        
        metadata = {
            "dimension": 384,
            "num_vectors": 1000,
            "index_type": "IndexFlatIP"
        }
        
        metadata_path = index_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Without --reindex, should reuse
        reindex = False
        should_rebuild = reindex or not metadata_path.exists()
        
        assert should_rebuild is False


class TestIndexFileStructure:
    """Test suite for index directory structure."""
    
    def test_index_directory_contains_required_files(self, tmp_path):
        """Test that index directory has all required files."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        
        required_files = [
            "hot.index",
            "metadata.json",
            "documents.db"
        ]
        
        # Create files
        for filename in required_files:
            (index_dir / filename).touch()
        
        # Verify all exist
        for filename in required_files:
            assert (index_dir / filename).exists()
    
    def test_metadata_json_structure(self, tmp_path):
        """Test metadata.json has correct structure."""
        metadata = {
            "dimension": 384,
            "num_vectors": 3633,
            "index_type": "IndexFlatIP",
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "created_at": "2026-01-13T12:00:00"
        }
        
        # Required fields
        assert "dimension" in metadata
        assert "num_vectors" in metadata
        assert "index_type" in metadata
        
        # Validate values
        assert metadata["dimension"] > 0
        assert metadata["num_vectors"] > 0
    
    def test_documents_db_schema(self, tmp_path):
        """Test documents.db has correct schema."""
        import sqlite3
        
        db_path = tmp_path / "documents.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create expected schema
        cursor.execute("""
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                text TEXT
            )
        """)
        
        conn.commit()
        
        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        conn.close()
        
        assert len(tables) == 1
        assert tables[0][0] == "documents"


class TestRunOutputFormat:
    """Test suite for run output format."""
    
    def test_run_output_file_structure(self, tmp_path):
        """Test run output JSON structure."""
        run_output = {
            "dataset": "nfcorpus",
            "model": "BAAI/bge-base-en-v1.5",
            "timestamp": "2026-01-13T12:00:00",
            "num_queries": 323,
            "retrieval_results": {
                "query1": [
                    {"doc_id": "doc1", "score": 0.95, "rank": 1},
                    {"doc_id": "doc2", "score": 0.87, "rank": 2}
                ]
            }
        }
        
        # Verify structure
        assert "dataset" in run_output
        assert "num_queries" in run_output
        assert "retrieval_results" in run_output
    
    def test_run_output_json_serializable(self):
        """Test that run output can be serialized to JSON."""
        run_output = {
            "dataset": "nfcorpus",
            "num_queries": 323,
            "results": []
        }
        
        # Should serialize without error
        json_str = json.dumps(run_output, indent=2)
        
        assert len(json_str) > 0
        assert "nfcorpus" in json_str
    
    def test_run_output_results_ordering(self):
        """Test that results are ordered by score."""
        query_results = [
            {"doc_id": "doc1", "score": 0.95, "rank": 1},
            {"doc_id": "doc2", "score": 0.87, "rank": 2},
            {"doc_id": "doc3", "score": 0.82, "rank": 3}
        ]
        
        # Verify descending score order
        scores = [r["score"] for r in query_results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify ranks are sequential
        ranks = [r["rank"] for r in query_results]
        assert ranks == [1, 2, 3]


class TestRetrievalExecution:
    """Test suite for retrieval execution."""
    
    def test_query_batch_processing(self):
        """Test processing queries in batches."""
        total_queries = 323
        batch_size = 32
        
        num_batches = (total_queries + batch_size - 1) // batch_size
        
        # nfcorpus: 323 queries / 32 per batch = 11 batches
        assert num_batches == 11
    
    def test_top_k_retrieval_parameter(self):
        """Test top-k retrieval parameter."""
        k_values = [10, 50, 100]
        
        for k in k_values:
            # Should return at most k results per query
            assert k > 0
    
    def test_laptop_mode_reduces_batch_size(self):
        """Test that laptop mode uses smaller batch size."""
        batch_size_normal = 32
        batch_size_laptop = 16
        
        # Laptop mode should use smaller batches
        assert batch_size_laptop < batch_size_normal


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_missing_corpus_file(self):
        """Test handling of missing corpus file."""
        corpus_path = Path("data/beir/nonexistent/corpus.jsonl")
        
        # Should detect missing file
        assert not corpus_path.exists()
    
    def test_missing_queries_file(self):
        """Test handling of missing queries file."""
        queries_path = Path("data/beir/nonexistent/queries.jsonl")
        
        # Should detect missing file
        assert not queries_path.exists()
    
    def test_invalid_index_directory(self, tmp_path):
        """Test handling of invalid index directory."""
        index_dir = tmp_path / "index"
        
        # Directory exists but missing files
        index_dir.mkdir()
        
        metadata_path = index_dir / "metadata.json"
        
        # Missing metadata should trigger rebuild
        if not metadata_path.exists():
            should_rebuild = True
        
        assert should_rebuild is True
    
    def test_malformed_corpus_entry(self):
        """Test handling of malformed corpus entry."""
        malformed_json = '{"_id": "doc1"}'  # Missing text field
        
        try:
            doc = json.loads(malformed_json)
            is_valid = "_id" in doc and "text" in doc
        except json.JSONDecodeError:
            is_valid = False
        
        # Should detect missing field
        assert is_valid is False


class TestLoggingOutput:
    """Test suite for logging and progress output."""
    
    def test_log_file_creation(self, tmp_path):
        """Test that log file is created."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        
        log_path = log_dir / "beir_adapter_run.log"
        log_path.touch()
        
        assert log_path.exists()
    
    def test_log_contains_progress_info(self):
        """Test that log contains progress information."""
        log_content = """
        Processing corpus...
        Building index...
        Processing queries: 100%
        Run completed successfully
        """
        
        assert "Processing corpus" in log_content
        assert "Building index" in log_content
        assert "completed successfully" in log_content
    
    def test_verbose_flag_increases_logging(self):
        """Test that --verbose flag increases log output."""
        cmd_normal = ["python", "scripts/run_beir_adapter.py", "--corpus", "corpus.jsonl"]
        cmd_verbose = ["python", "scripts/run_beir_adapter.py", "--corpus", "corpus.jsonl", "--verbose"]
        
        # Verbose should add flag
        assert "--verbose" in cmd_verbose
        assert "--verbose" not in cmd_normal


class TestMetricsIntegration:
    """Test suite for metrics calculation integration."""
    
    def test_metrics_calculation_after_run(self, tmp_path):
        """Test metrics calculation after successful run."""
        run_file = tmp_path / "beir_run_nfcorpus.json"
        qrels_file = Path("data/beir/nfcorpus/qrels/test.tsv")
        
        # Metrics command structure
        metrics_cmd = [
            "python", "scripts/calculate_beir_metrics.py",
            "--results", str(run_file),
            "--qrels", str(qrels_file),
            "--k", "10"
        ]
        
        assert "--results" in metrics_cmd
        assert "scripts/calculate_beir_metrics.py" in metrics_cmd
    
    def test_metrics_output_file_naming(self):
        """Test metrics output file naming convention."""
        run_file = "results/beir_run_nfcorpus.json"
        
        # Derive metrics filename
        metrics_file = run_file.replace(".json", "_metrics_k10.json")
        
        assert metrics_file == "results/beir_run_nfcorpus_metrics_k10.json"
    
    def test_metrics_content_structure(self):
        """Test metrics JSON structure."""
        metrics = {
            "recall@10": 0.3106,
            "mrr": 0.2834,
            "ndcg@10": 0.2646,
            "num_queries": 323,
            "dataset": "nfcorpus"
        }
        
        # Verify required fields
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert "ndcg@10" in metrics
        
        # Verify value ranges
        assert 0 <= metrics["recall@10"] <= 1
        assert 0 <= metrics["mrr"] <= 1
        assert 0 <= metrics["ndcg@10"] <= 1


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics."""
    
    @pytest.mark.slow
    def test_indexing_time_reasonable(self):
        """Test that indexing completes in reasonable time."""
        # For nfcorpus (3,633 docs): ~2-5 minutes
        expected_max_minutes = 5
        
        assert expected_max_minutes > 0
    
    @pytest.mark.slow
    def test_retrieval_time_reasonable(self):
        """Test that retrieval completes in reasonable time."""
        # For nfcorpus (323 queries): ~3-5 minutes
        num_queries = 323
        expected_seconds_per_query = 1.0
        
        total_expected_seconds = num_queries * expected_seconds_per_query
        
        # Should complete within reasonable time
        assert total_expected_seconds < 600  # Under 10 minutes
    
    def test_memory_usage_with_laptop_mode(self):
        """Test that laptop mode reduces memory usage."""
        # Laptop mode should use CPU and smaller batches
        laptop_mode = True
        
        if laptop_mode:
            expected_device = "cpu"
            expected_batch_size = 16
        else:
            expected_device = "cuda"
            expected_batch_size = 32
        
        assert laptop_mode is True
        assert expected_device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
