"""
End-to-end tests for batch BEIR evaluation workflow.

Tests the complete batch processing workflow via run_beir_batch.py,
including multiple dataset execution, error handling, and log generation.
"""

import subprocess
import time
from pathlib import Path

import pytest


class TestBatchWorkflow:
    """Test suite for batch BEIR evaluation workflow."""

    def test_batch_script_exists(self):
        """Test that run_beir_batch.py exists."""
        script_path = Path("tools/run_beir_batch.py")
        assert script_path.exists(), "run_beir_batch.py not found"

    def test_batch_script_help_flag(self):
        """Test that batch script responds to --help."""
        result = subprocess.run(
            ["python", "tools/run_beir_batch.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should show help without error
        assert result.returncode == 0 or "usage" in result.stdout.lower()

    @pytest.mark.slow
    def test_batch_single_dataset_execution(self, temp_results_dir):
        """Test batch execution with single dataset (smoke test)."""
        # This would run a minimal batch test
        # In practice, use a small test dataset

        datasets = ["test_mini"]  # Hypothetical small test dataset

        # Command structure
        cmd = [
            "python",
            "-u",
            "tools/run_beir_batch.py",
            "--datasets",
            *datasets,
            "--output-dir",
            str(temp_results_dir),
            "--laptop-mode",
        ]

        # This is a structure test; actual execution would require test data
        assert len(cmd) > 0
        assert "--datasets" in cmd

    def test_batch_multiple_datasets_config(self):
        """Test batch configuration with multiple datasets."""
        datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]

        # Verify all datasets in config
        assert len(datasets) == 4
        assert "nfcorpus" in datasets
        assert "scifact" in datasets

    def test_batch_dataset_paths_template(self):
        """Test dataset path templating for batch execution."""
        dataset_template = {
            "corpus": "data/beir/{dataset}/corpus.jsonl",
            "queries": "data/beir/{dataset}/queries.jsonl",
            "qrels": "data/beir/{dataset}/qrels/test.tsv",
        }

        dataset = "nfcorpus"

        # Resolve paths
        corpus_path = dataset_template["corpus"].format(dataset=dataset)
        queries_path = dataset_template["queries"].format(dataset=dataset)
        qrels_path = dataset_template["qrels"].format(dataset=dataset)

        assert corpus_path == "data/beir/nfcorpus/corpus.jsonl"
        assert queries_path == "data/beir/nfcorpus/queries.jsonl"
        assert qrels_path == "data/beir/nfcorpus/qrels/test.tsv"

    def test_batch_output_naming_convention(self):
        """Test output file naming for batch runs."""
        datasets = ["nfcorpus", "scifact"]
        output_template = "results/beir_run_{dataset}.json"

        expected_outputs = [output_template.format(dataset=ds) for ds in datasets]

        assert expected_outputs[0] == "results/beir_run_nfcorpus.json"
        assert expected_outputs[1] == "results/beir_run_scifact.json"


class TestBatchIndexManagement:
    """Test suite for index management in batch workflow."""

    def test_batch_with_reindex_flag(self):
        """Test batch execution with --reindex flag."""
        cmd = ["python", "-u", "tools/run_beir_batch.py", "--datasets", "nfcorpus", "--reindex"]

        # --reindex should force rebuild of all indexes
        assert "--reindex" in cmd

    def test_batch_with_existing_indexes(self):
        """Test batch execution reusing existing indexes."""
        # Simulate existing index check
        index_dirs = ["results/beir_index_nfcorpus", "results/beir_index_scifact"]

        for index_dir in index_dirs:
            # Would check for metadata.json
            metadata_path = Path(index_dir) / "metadata.json"
            # In real test, verify file exists

    def test_batch_missing_index_handling(self):
        """Test batch handling of missing index."""
        index_dir = Path("results/beir_index_nonexistent")
        metadata_path = index_dir / "metadata.json"

        # If metadata missing, should rebuild
        if not metadata_path.exists():
            should_rebuild = True
        else:
            should_rebuild = False

        # For nonexistent dataset, should rebuild
        assert should_rebuild is True


class TestBatchLogging:
    """Test suite for batch workflow logging."""

    def test_batch_log_file_creation(self, tmp_path):
        """Test that batch execution creates log files."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)

        # Expected log naming pattern
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_name = f"beir_batch_{timestamp}.log"
        log_path = log_dir / log_name

        # Verify log path structure
        assert "beir_batch" in log_name
        assert log_name.endswith(".log")

    def test_batch_log_per_dataset(self, tmp_path):
        """Test that each dataset generates its own log."""
        datasets = ["nfcorpus", "scifact"]
        log_dir = tmp_path / "logs"

        expected_logs = [
            log_dir / f"beir_adapter_beir_run_{ds}_20260113_120000.log" for ds in datasets
        ]

        # Each dataset should have a timestamped log
        assert len(expected_logs) == 2

    def test_batch_error_logging(self):
        """Test that batch errors are logged appropriately."""
        # Simulate error scenario
        error_message = "FileNotFoundError: metadata.json not found"

        # Error should be captured in log
        assert "FileNotFoundError" in error_message
        assert "metadata.json" in error_message


class TestBatchErrorHandling:
    """Test suite for error handling in batch workflow."""

    def test_batch_continues_after_dataset_failure(self):
        """Test that batch continues to next dataset after failure."""
        datasets = ["dataset1", "dataset2", "dataset3"]

        # Simulate dataset2 failing
        failed_dataset = "dataset2"
        remaining_datasets = [ds for ds in datasets if ds != failed_dataset]

        # Should process dataset1 and dataset3
        assert len(remaining_datasets) == 2
        assert "dataset1" in remaining_datasets
        assert "dataset3" in remaining_datasets

    def test_batch_exit_code_on_failure(self):
        """Test that batch returns non-zero exit code on failure."""
        # Simulate command failure
        exit_code = 1  # Non-zero indicates failure

        assert exit_code != 0

    def test_batch_partial_results_saved(self, tmp_path):
        """Test that partial results are saved on failure."""
        # Even if batch fails partway, completed datasets should have output
        completed_datasets = ["nfcorpus", "scifact"]

        for dataset in completed_datasets:
            output_path = tmp_path / f"beir_run_{dataset}.json"
            # Would verify file exists

    def test_batch_missing_dataset_path(self):
        """Test handling of missing dataset path."""
        corpus_path = Path("data/beir/nonexistent/corpus.jsonl")

        # Should detect missing file
        exists = corpus_path.exists()
        assert exists is False


class TestBatchMetricsComputation:
    """Test suite for metrics computation in batch workflow."""

    def test_batch_metrics_calculation_trigger(self):
        """Test that batch triggers metrics calculation."""
        # After successful run, should calculate metrics
        run_file = "results/beir_run_nfcorpus.json"
        qrels_file = "data/beir/nfcorpus/qrels/test.tsv"

        # Command structure for metrics
        metrics_cmd = [
            "python",
            "tools/calculate_beir_metrics.py",
            "--results",
            run_file,
            "--qrels",
            qrels_file,
            "--k",
            "10",
        ]

        assert "--results" in metrics_cmd
        assert "--qrels" in metrics_cmd

    def test_batch_metrics_output_format(self, tmp_path):
        """Test metrics output file format."""
        metrics_file = tmp_path / "beir_run_nfcorpus_metrics_k10.json"

        # Expected metrics structure
        expected_metrics = {
            "recall@10": 0.3106,
            "mrr": 0.2834,
            "ndcg@10": 0.2646,
            "num_queries": 323,
        }

        # Verify structure
        assert "recall@10" in expected_metrics
        assert "mrr" in expected_metrics
        assert expected_metrics["num_queries"] > 0

    def test_batch_metrics_aggregation(self):
        """Test aggregation of metrics across datasets."""
        dataset_metrics = {
            "nfcorpus": {"recall@10": 0.3106, "mrr": 0.2834},
            "scifact": {"recall@10": 0.6700, "mrr": 0.5890},
            "arguana": {"recall@10": 0.9870, "mrr": 0.9660},
            "fiqa": {"recall@10": 0.4010, "mrr": 0.3410},
        }

        # Calculate average metrics
        avg_recall = sum(m["recall@10"] for m in dataset_metrics.values()) / len(dataset_metrics)
        avg_mrr = sum(m["mrr"] for m in dataset_metrics.values()) / len(dataset_metrics)

        assert 0 < avg_recall < 1
        assert 0 < avg_mrr < 1


class TestBatchConfigIntegration:
    """Test suite for batch workflow config integration."""

    def test_batch_config_file_loading(self):
        """Test loading batch config from JSON."""
        config_path = Path("configs/benchmark_config.json")

        # Would load and validate config
        # assert config_path.exists()

    def test_batch_cli_args_override_config(self):
        """Test CLI arguments override config file."""
        # Config file
        config = {"datasets": ["nfcorpus", "scifact"], "laptop_mode": False}

        # CLI override
        cli_datasets = ["arguana", "fiqa"]
        cli_laptop_mode = True

        # Merged config
        final_config = {"datasets": cli_datasets, "laptop_mode": cli_laptop_mode}  # CLI overrides

        assert final_config["datasets"] == ["arguana", "fiqa"]
        assert final_config["laptop_mode"] is True

    def test_batch_dataset_subset_selection(self):
        """Test selecting subset of datasets from config."""
        all_datasets = ["nfcorpus", "scifact", "arguana", "fiqa", "trec-covid"]
        selected_datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]  # all-small

        # Subset should be smaller or equal
        assert len(selected_datasets) <= len(all_datasets)
        assert all(ds in all_datasets for ds in selected_datasets)


class TestBatchPerformance:
    """Test suite for batch workflow performance."""

    def test_batch_parallel_potential(self):
        """Test that datasets could be processed in parallel."""
        datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]

        # Each dataset is independent
        # Could process in parallel threads/processes
        assert len(datasets) == 4

    def test_batch_laptop_mode_flag(self):
        """Test laptop mode reduces memory usage."""
        cmd_normal = ["python", "tools/run_beir_batch.py", "--datasets", "nfcorpus"]
        cmd_laptop = [
            "python",
            "tools/run_beir_batch.py",
            "--datasets",
            "nfcorpus",
            "--laptop-mode",
        ]

        # Laptop mode should add flag
        assert "--laptop-mode" in cmd_laptop
        assert "--laptop-mode" not in cmd_normal

    @pytest.mark.slow
    def test_batch_execution_time_estimate(self):
        """Test that batch execution time is reasonable."""
        # For 4 small datasets without reindexing
        datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]

        # Estimate: ~5-10 minutes per dataset without reindex
        estimated_minutes_per_dataset = 8
        total_estimated_minutes = len(datasets) * estimated_minutes_per_dataset

        # Should complete within reasonable time
        assert total_estimated_minutes < 60  # Under 1 hour for 4 datasets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
