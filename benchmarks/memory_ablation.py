#!/usr/bin/env python3
"""
Memory Ablation Benchmark for CUBO Paper.

Measures exact RAM savings from each optimization:
- Memory-mapped embeddings (expected: 80-90% savings)
- 8-bit quantization (expected: 4x reduction)
- IVF+PQ indexing (expected: 8x additional compression)
- Lazy model loading (expected: 300-800 MB savings)
- Semantic cache (expected: 50-200 MB)

Output: JSON report suitable for paper ablation table.

Usage:
python benchmarks/memory_ablation.py --data-folder data/ultradomain_small --output results/memory_ablation.json
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
except ImportError:
    psutil = None

from benchmarks.utils.hardware import log_hardware_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryAblationBenchmark:
    """Benchmark memory usage under different configurations."""

    def __init__(self, data_folder: str, num_queries: int = 50):
        self.data_folder = Path(data_folder)
        self.num_queries = num_queries
        self.hardware = log_hardware_metadata()
        self.results = {
            "metadata": {
                "timestamp": time.time(),
                "data_folder": str(data_folder),
                "num_queries": num_queries,
                "hardware": self.hardware,
            },
            "configurations": [],
        }

    def _get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        if psutil:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 3)
        return 0.0

    def _get_system_memory_gb(self) -> Dict[str, float]:
        """Get system memory stats."""
        if psutil:
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "used_gb": mem.used / (1024 ** 3),
                "percent": mem.percent,
            }
        return {}

    def _force_gc(self):
        """Force garbage collection."""
        gc.collect()
        time.sleep(0.5)
        gc.collect()

    def measure_configuration(
        self,
        config_name: str,
        config_updates: Dict[str, Any],
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Measure memory usage for a specific configuration.
        
        Returns dict with memory metrics and timing.
        """
        logger.info(f"Measuring configuration: {config_name}")
        self._force_gc()
        
        # Baseline memory
        mem_before = self._get_memory_usage_gb()
        sys_mem_before = self._get_system_memory_gb()
        
        result = {
            "name": config_name,
            "description": description,
            "config_updates": config_updates,
            "memory_before_gb": mem_before,
            "system_memory_before": sys_mem_before,
        }
        
        try:
            from src.cubo.config import config
            
            # Apply configuration
            for key, value in config_updates.items():
                if isinstance(value, dict):
                    current = config.get(key, {})
                    if isinstance(current, dict):
                        current.update(value)
                        config.set(key, current)
                    else:
                        config.set(key, value)
                else:
                    config.set(key, value)
            
            # Import and initialize CUBO
            start_time = time.time()
            
            from src.cubo.main import CUBOApp
            cubo = CUBOApp()
            
            init_time = time.time()
            if not cubo.initialize_components():
                result["error"] = "Failed to initialize components"
                return result
            
            component_init_time = time.time() - init_time
            
            # Memory after component initialization
            mem_after_init = self._get_memory_usage_gb()
            
            # Load and index documents
            docs = cubo.doc_loader.load_documents_from_folder(str(self.data_folder))
            if not docs:
                result["error"] = "No documents loaded"
                return result
            
            doc_texts = [d["text"] if isinstance(d, dict) else str(d) for d in docs]
            cubo.retriever.add_documents(doc_texts[:min(len(doc_texts), 5000)])  # Cap for speed
            
            mem_after_index = self._get_memory_usage_gb()
            
            # Run queries
            query_latencies = []
            test_queries = [
                "What is the main topic?",
                "Explain the key concepts.",
                "What are the conclusions?",
            ] * (self.num_queries // 3 + 1)
            
            for query in test_queries[:self.num_queries]:
                q_start = time.time()
                _ = cubo.retriever.retrieve_top_documents(query, top_k=10)
                query_latencies.append((time.time() - q_start) * 1000)
            
            mem_after_queries = self._get_memory_usage_gb()
            peak_memory = max(mem_after_init, mem_after_index, mem_after_queries)
            
            total_time = time.time() - start_time
            
            result.update({
                "success": True,
                "documents_indexed": min(len(doc_texts), 5000),
                "queries_run": self.num_queries,
                "memory_after_init_gb": mem_after_init,
                "memory_after_index_gb": mem_after_index,
                "memory_after_queries_gb": mem_after_queries,
                "peak_memory_gb": peak_memory,
                "memory_delta_gb": peak_memory - mem_before,
                "component_init_time_s": component_init_time,
                "total_time_s": total_time,
                "avg_query_latency_ms": sum(query_latencies) / len(query_latencies) if query_latencies else 0,
                "p50_query_latency_ms": sorted(query_latencies)[len(query_latencies) // 2] if query_latencies else 0,
                "p95_query_latency_ms": sorted(query_latencies)[int(len(query_latencies) * 0.95)] if query_latencies else 0,
                "fits_16gb": peak_memory < 15.5,  # Leave room for OS
            })
            
            # Cleanup
            del cubo
            self._force_gc()
            
        except Exception as e:
            logger.error(f"Configuration {config_name} failed: {e}", exc_info=True)
            result["error"] = str(e)
            result["success"] = False
        
        return result

    def run_all_ablations(self) -> Dict[str, Any]:
        """Run all ablation configurations."""
        
        configurations = [
            {
                "name": "full_cubo_laptop",
                "description": "Full CUBO with all laptop-mode optimizations",
                "config_updates": {
                    "laptop_mode": True,
                    "embedding_storage": "mmap",
                }
            },
            {
                "name": "no_mmap",
                "description": "Disable memory-mapped embeddings",
                "config_updates": {
                    "laptop_mode": True,
                    "embedding_storage": "memory",
                }
            },
            {
                "name": "no_8bit",
                "description": "Disable 8-bit quantization (use fp32)",
                "config_updates": {
                    "laptop_mode": True,
                    "embedding_quantization": "fp32",
                }
            },
            {
                "name": "flat_index",
                "description": "Use flat FAISS index instead of IVF+PQ",
                "config_updates": {
                    "laptop_mode": True,
                    "faiss_index_type": "flat",
                }
            },
        ]
        
        for cfg in configurations:
            result = self.measure_configuration(
                cfg["name"],
                cfg["config_updates"],
                cfg["description"],
            )
            self.results["configurations"].append(result)
            
            # Log summary
            if result.get("success"):
                logger.info(
                    f"  {cfg['name']}: peak={result['peak_memory_gb']:.2f} GB, "
                    f"delta={result['memory_delta_gb']:.2f} GB, "
                    f"fits_16gb={result['fits_16gb']}"
                )
            else:
                logger.warning(f"  {cfg['name']}: FAILED - {result.get('error')}")
        
        # Calculate savings
        self._calculate_savings()
        
        return self.results

    def _calculate_savings(self):
        """Calculate memory savings between configurations."""
        configs = {c["name"]: c for c in self.results["configurations"] if c.get("success")}
        
        savings = {}
        
        if "full_cubo_laptop" in configs and "no_mmap" in configs:
            baseline = configs["no_mmap"]["peak_memory_gb"]
            optimized = configs["full_cubo_laptop"]["peak_memory_gb"]
            savings["mmap_savings_gb"] = baseline - optimized
            savings["mmap_savings_percent"] = ((baseline - optimized) / baseline) * 100 if baseline > 0 else 0
        
        if "full_cubo_laptop" in configs and "no_8bit" in configs:
            baseline = configs["no_8bit"]["peak_memory_gb"]
            optimized = configs["full_cubo_laptop"]["peak_memory_gb"]
            savings["quantization_savings_gb"] = baseline - optimized
        
        if "full_cubo_laptop" in configs and "flat_index" in configs:
            baseline = configs["flat_index"]["peak_memory_gb"]
            optimized = configs["full_cubo_laptop"]["peak_memory_gb"]
            savings["ivf_pq_savings_gb"] = baseline - optimized
        
        self.results["savings_summary"] = savings

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print ablation summary table."""
        print("\n" + "=" * 80)
        print("MEMORY ABLATION BENCHMARK RESULTS")
        print("=" * 80)
        
        print(f"\nHardware: {self.hardware['cpu']['model']}, {self.hardware['ram']['total_gb']:.1f} GB RAM")
        print(f"Data folder: {self.data_folder}")
        
        print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
            "Configuration", "Peak (GB)", "Delta (GB)", "Latency (ms)", "Fits 16GB?"
        ))
        print("-" * 75)
        
        for cfg in self.results["configurations"]:
            if cfg.get("success"):
                print("{:<25} {:>12.2f} {:>12.2f} {:>12.1f} {:>10}".format(
                    cfg["name"],
                    cfg["peak_memory_gb"],
                    cfg["memory_delta_gb"],
                    cfg["avg_query_latency_ms"],
                    "Yes" if cfg["fits_16gb"] else "No"
                ))
            else:
                print("{:<25} {:>12} {:>12} {:>12} {:>10}".format(
                    cfg["name"], "FAILED", "-", "-", "-"
                ))
        
        if "savings_summary" in self.results:
            print("\nSavings Summary:")
            for key, value in self.results["savings_summary"].items():
                if "percent" in key:
                    print(f"  {key}: {value:.1f}%")
                else:
                    print(f"  {key}: {value:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="CUBO Memory Ablation Benchmark")
    parser.add_argument("--data-folder", required=True, help="Path to test data")
    parser.add_argument("--output", default="results/memory_ablation.json", help="Output JSON path")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to run")
    
    args = parser.parse_args()
    
    benchmark = MemoryAblationBenchmark(args.data_folder, args.num_queries)
    benchmark.run_all_ablations()
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
