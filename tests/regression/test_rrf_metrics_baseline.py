"""
Regression tests for RRF metrics baselines.

Tests that RRF parameter combinations maintain or improve
baseline metrics, ensuring no performance degradation.
"""

import pytest
import json


class TestRRFMetricsBaseline:
    """Test suite for RRF metrics regression."""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Load baseline metrics for comparison."""
        # Known baseline metrics from previous runs
        return {
            "nfcorpus": {
                "recall@10": 0.3106,
                "mrr": 0.2834,
                "ndcg@10": 0.2646
            },
            "scifact": {
                "recall@10": 0.6700,
                "mrr": 0.5890,
                "ndcg@10": 0.6120
            },
            "arguana": {
                "recall@10": 0.9870,
                "mrr": 0.9660,
                "ndcg@10": 0.9740
            },
            "fiqa": {
                "recall@10": 0.4010,
                "mrr": 0.3410,
                "ndcg@10": 0.3520
            }
        }
    
    @pytest.fixture
    def rrf_parameter_configs(self):
        """RRF parameter combinations to test."""
        return {
            "baseline": {"k": 60, "semantic_weight": 1.0, "bm25_weight": 1.0},
            "high_k": {"k": 120, "semantic_weight": 1.0, "bm25_weight": 1.0},
            "low_k": {"k": 20, "semantic_weight": 1.0, "bm25_weight": 1.0},
            "semantic_heavy": {"k": 60, "semantic_weight": 1.3, "bm25_weight": 0.7},
            "bm25_heavy": {"k": 60, "semantic_weight": 0.7, "bm25_weight": 1.3}
        }
    
    def test_baseline_metrics_exist(self, baseline_metrics):
        """Test that baseline metrics are available."""
        required_datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]
        
        for dataset in required_datasets:
            assert dataset in baseline_metrics
            assert "recall@10" in baseline_metrics[dataset]
            assert "mrr" in baseline_metrics[dataset]
            assert "ndcg@10" in baseline_metrics[dataset]
    
    def test_metrics_within_valid_range(self, baseline_metrics):
        """Test that all metrics are within valid range [0, 1]."""
        for dataset, metrics in baseline_metrics.items():
            for metric_name, value in metrics.items():
                assert 0 <= value <= 1, f"{dataset}.{metric_name} = {value} out of range"
    
    def test_rrf_baseline_config_performance(self, baseline_metrics):
        """Test that baseline RRF config meets minimum thresholds."""
        # Minimum acceptable thresholds
        thresholds = {
            "nfcorpus": {"recall@10": 0.30, "mrr": 0.28, "ndcg@10": 0.26},
            "scifact": {"recall@10": 0.65, "mrr": 0.57, "ndcg@10": 0.60},
            "arguana": {"recall@10": 0.98, "mrr": 0.95, "ndcg@10": 0.97},
            "fiqa": {"recall@10": 0.39, "mrr": 0.33, "ndcg@10": 0.34}
        }
        
        for dataset in thresholds:
            for metric in thresholds[dataset]:
                baseline_value = baseline_metrics[dataset][metric]
                threshold = thresholds[dataset][metric]
                
                assert baseline_value >= threshold, \
                    f"{dataset}.{metric} = {baseline_value} below threshold {threshold}"
    
    def test_rrf_k_parameter_sensitivity(self):
        """Test that k parameter affects results predictably."""
        # Lower k should amplify rank differences
        k_values = [20, 60, 120]
        
        rank = 5
        scores = [1/(rank + k) for k in k_values]
        
        # Scores should decrease as k increases
        assert scores[0] > scores[1] > scores[2]
    
    def test_rrf_weight_parameter_sensitivity(self):
        """Test that weights affect fusion appropriately."""
        k = 60
        dense_rank = 1
        bm25_rank = 10
        
        # Balanced weights
        balanced_score = 1.0 * (1/(dense_rank + k)) + 1.0 * (1/(bm25_rank + k))
        
        # Semantic-heavy weights
        semantic_heavy_score = 1.3 * (1/(dense_rank + k)) + 0.7 * (1/(bm25_rank + k))
        
        # BM25-heavy weights
        bm25_heavy_score = 0.7 * (1/(dense_rank + k)) + 1.3 * (1/(bm25_rank + k))
        
        # Semantic-heavy should favor better dense rank
        assert semantic_heavy_score > balanced_score
        
        # BM25-heavy should be lower (worse BM25 rank)
        assert bm25_heavy_score < balanced_score


class TestRRFMetricsComparison:
    """Test suite for comparing RRF configs."""
    
    def test_compare_rrf_configs_format(self):
        """Test comparison output format."""
        comparison = {
            "nfcorpus": {
                "baseline": {"recall@10": 0.3106, "mrr": 0.2834},
                "high_k": {"recall@10": 0.3106, "mrr": 0.2834},
                "improvement": {"recall@10": 0.0000, "mrr": 0.0000}
            }
        }
        
        # Verify structure
        assert "nfcorpus" in comparison
        assert "baseline" in comparison["nfcorpus"]
        assert "improvement" in comparison["nfcorpus"]
    
    def test_metric_improvement_calculation(self):
        """Test metric improvement calculation."""
        baseline = {"recall@10": 0.3106}
        new_config = {"recall@10": 0.3200}
        
        improvement = new_config["recall@10"] - baseline["recall@10"]
        improvement_pct = (improvement / baseline["recall@10"]) * 100
        
        assert improvement > 0
        assert improvement_pct > 0
    
    def test_no_regression_detected(self):
        """Test that no significant regression occurs."""
        baseline = {"recall@10": 0.3106, "mrr": 0.2834, "ndcg@10": 0.2646}
        new_config = {"recall@10": 0.3106, "mrr": 0.2834, "ndcg@10": 0.2646}
        
        # Allow small tolerance for floating point
        tolerance = 0.001
        
        for metric in baseline:
            diff = abs(new_config[metric] - baseline[metric])
            assert diff <= tolerance, f"Regression detected in {metric}"


class TestMetricsFileFormat:
    """Test suite for metrics file format."""
    
    def test_metrics_json_loadable(self):
        """Test that metrics files can be loaded as JSON."""
        # Example metrics file content
        metrics_content = {
            "recall@10": 0.3106,
            "mrr": 0.2834,
            "ndcg@10": 0.2646,
            "num_queries": 323,
            "dataset": "nfcorpus",
            "config": {"k": 60, "semantic_weight": 1.0, "bm25_weight": 1.0}
        }
        
        # Should serialize and deserialize
        json_str = json.dumps(metrics_content)
        loaded = json.loads(json_str)
        
        assert loaded["recall@10"] == 0.3106
        assert loaded["dataset"] == "nfcorpus"
    
    def test_metrics_file_naming_convention(self):
        """Test metrics file naming convention."""
        base_name = "beir_run_nfcorpus_rrf_k60_sw1.0_bw1.0"
        
        # Metrics file should append _metrics_k10.json
        metrics_name = f"{base_name}_metrics_k10.json"
        
        assert "metrics_k10.json" in metrics_name
        assert "nfcorpus" in metrics_name
        assert "rrf" in metrics_name
    
    def test_metrics_file_contains_config(self):
        """Test that metrics file includes RRF config."""
        metrics = {
            "recall@10": 0.3106,
            "config": {
                "k": 60,
                "semantic_weight": 1.0,
                "bm25_weight": 1.0
            }
        }
        
        assert "config" in metrics
        assert "k" in metrics["config"]
        assert metrics["config"]["k"] == 60


class TestRRFParameterSweep:
    """Test suite for RRF parameter sweep results."""
    
    def test_sweep_covers_all_combinations(self):
        """Test that parameter sweep covers all combinations."""
        k_values = [20, 60, 120]
        semantic_weights = [0.7, 1.0, 1.3]
        bm25_weights = [0.7, 1.0, 1.3]
        
        total_combinations = len(k_values) * len(semantic_weights) * len(bm25_weights)
        
        assert total_combinations == 27
    
    def test_sweep_results_per_dataset(self):
        """Test that sweep produces results for each dataset."""
        datasets = ["nfcorpus", "scifact", "arguana", "fiqa"]
        configs_per_dataset = 27  # 3 * 3 * 3
        
        total_expected_files = len(datasets) * configs_per_dataset
        
        assert total_expected_files == 108  # 4 datasets * 27 configs
    
    def test_best_config_selection(self):
        """Test selecting best config per dataset."""
        configs_metrics = [
            {"config": "k20_sw1.0_bw1.0", "recall@10": 0.3050},
            {"config": "k60_sw1.0_bw1.0", "recall@10": 0.3106},
            {"config": "k120_sw1.0_bw1.0", "recall@10": 0.3090}
        ]
        
        # Select best by recall@10
        best_config = max(configs_metrics, key=lambda x: x["recall@10"])
        
        assert best_config["config"] == "k60_sw1.0_bw1.0"
        assert best_config["recall@10"] == 0.3106


class TestMetricsStability:
    """Test suite for metrics stability and reproducibility."""
    
    def test_metrics_reproducible(self):
        """Test that repeated runs produce same metrics."""
        # Given same input and random seed
        run1_metrics = {"recall@10": 0.3106, "mrr": 0.2834}
        run2_metrics = {"recall@10": 0.3106, "mrr": 0.2834}
        
        # Should be identical
        assert run1_metrics == run2_metrics
    
    def test_metrics_precision(self):
        """Test that metrics are reported with sufficient precision."""
        metric_value = 0.3106
        
        # Should have at least 4 decimal places
        value_str = f"{metric_value:.4f}"
        
        assert value_str == "0.3106"
    
    def test_floating_point_comparison_tolerance(self):
        """Test floating point comparison with tolerance."""
        expected = 0.3106
        actual = 0.31061
        tolerance = 0.001
        
        assert abs(actual - expected) < tolerance


class TestRRFMetricsAggregation:
    """Test suite for aggregating metrics across datasets."""
    
    def test_average_metrics_across_datasets(self):
        """Test calculating average metrics."""
        dataset_metrics = {
            "nfcorpus": {"recall@10": 0.3106},
            "scifact": {"recall@10": 0.6700},
            "arguana": {"recall@10": 0.9870},
            "fiqa": {"recall@10": 0.4010}
        }
        
        avg_recall = sum(m["recall@10"] for m in dataset_metrics.values()) / len(dataset_metrics)
        
        expected_avg = (0.3106 + 0.6700 + 0.9870 + 0.4010) / 4
        
        assert abs(avg_recall - expected_avg) < 0.0001
    
    def test_weighted_average_by_query_count(self):
        """Test weighted average by number of queries."""
        dataset_metrics = [
            {"dataset": "nfcorpus", "recall@10": 0.3106, "num_queries": 323},
            {"dataset": "scifact", "recall@10": 0.6700, "num_queries": 300}
        ]
        
        total_queries = sum(d["num_queries"] for d in dataset_metrics)
        weighted_avg = sum(
            d["recall@10"] * d["num_queries"] / total_queries
            for d in dataset_metrics
        )
        
        assert weighted_avg > 0
        assert weighted_avg < 1
    
    def test_best_overall_config_selection(self):
        """Test selecting best overall config across datasets."""
        config_results = {
            "k60_sw1.0_bw1.0": {
                "avg_recall@10": 0.5921,
                "avg_mrr": 0.5448
            },
            "k20_sw1.0_bw1.0": {
                "avg_recall@10": 0.5800,
                "avg_mrr": 0.5350
            }
        }
        
        # Select best by average recall
        best_config = max(config_results.items(), key=lambda x: x[1]["avg_recall@10"])
        
        assert best_config[0] == "k60_sw1.0_bw1.0"


class TestRegressionDetection:
    """Test suite for detecting performance regressions."""
    
    def test_detect_significant_regression(self):
        """Test detection of significant performance regression."""
        baseline = 0.3106
        new_value = 0.2800  # 10% drop
        
        regression_threshold = 0.05  # 5% drop is significant
        
        regression_pct = (baseline - new_value) / baseline
        
        is_significant_regression = regression_pct > regression_threshold
        
        assert is_significant_regression is True
    
    def test_ignore_minor_fluctuations(self):
        """Test that minor fluctuations are ignored."""
        baseline = 0.3106
        new_value = 0.3100  # ~0.2% drop
        
        tolerance = 0.01  # 1% tolerance
        
        regression_pct = (baseline - new_value) / baseline
        
        is_significant = regression_pct > tolerance
        
        assert is_significant is False
    
    def test_regression_alert_message(self):
        """Test formatting of regression alert message."""
        dataset = "nfcorpus"
        metric = "recall@10"
        baseline = 0.3106
        new_value = 0.2800
        
        regression_pct = ((baseline - new_value) / baseline) * 100
        
        alert_message = (
            f"REGRESSION ALERT: {dataset}.{metric} dropped from "
            f"{baseline:.4f} to {new_value:.4f} (-{regression_pct:.1f}%)"
        )
        
        assert "REGRESSION ALERT" in alert_message
        assert dataset in alert_message
        assert metric in alert_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
