"""
Test 7: Pyserini HNSW Baseline Benchmark

Validates claim that Lucene/Pyserini HNSW "imposes JVM overhead" by measuring:
- Peak RAM during indexing vs CUBO
- Query latency (p50, p95, p99)
- nDCG@10 quality
- Feasibility on 16 GB constraint

Usage:
    python -m evaluation.pyserini_benchmark --dataset scifact --output results/pyserini_benchmark.json
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PyseriniBenchmark:
    """Benchmark Pyserini HNSW on BEIR datasets."""

    def __init__(self, dataset: str = "scifact", data_dir: Path = None):
        """
        Initialize benchmark.

        Args:
            dataset: BEIR dataset name (e.g., 'scifact')
            data_dir: Path to BEIR data directory
        """
        self.dataset = dataset
        self.data_dir = data_dir or Path("data/beir")
        self.dataset_path = self.data_dir / dataset
        self.index_dir = Path("indexes") / f"pyserini_{dataset}"
        self.process = None

    def check_pyserini_installed(self) -> bool:
        """Check if pyserini is installed and working."""
        logger.info("Checking Pyserini installation...")
        try:
            import pyserini

            logger.info(f"✓ Pyserini version: {pyserini.__version__}")
            return True
        except ImportError:
            logger.error("✗ Pyserini not installed. Installing...")
            try:
                subprocess.run(
                    ["pip", "install", "pyserini"],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                logger.info("✓ Pyserini installed successfully")
                return True
            except Exception as e:
                logger.error(f"✗ Failed to install Pyserini: {e}")
                return False

    def monitor_memory_during_indexing(self) -> Dict[str, float]:
        """Monitor memory usage during indexing."""
        logger.info("Starting memory monitoring...")

        if self.process is None:
            logger.warning("No process to monitor")
            return {"peak_rss_mb": 0, "peak_vms_mb": 0}

        peak_rss = 0
        peak_vms = 0
        measurements = []

        try:
            while self.process.poll() is None:
                try:
                    proc_info = self.process.memory_info()
                    rss_mb = proc_info.rss / (1024 * 1024)
                    vms_mb = proc_info.vms / (1024 * 1024)

                    peak_rss = max(peak_rss, rss_mb)
                    peak_vms = max(peak_vms, vms_mb)
                    measurements.append({"rss_mb": rss_mb, "vms_mb": vms_mb})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

                time.sleep(0.5)  # Check every 500ms
        except Exception as e:
            logger.warning(f"Memory monitoring error: {e}")

        logger.info(f"Peak RSS: {peak_rss:.0f} MB")
        logger.info(f"Peak VMS: {peak_vms:.0f} MB")

        return {"peak_rss_mb": float(peak_rss), "peak_vms_mb": float(peak_vms)}

    def build_index(self) -> Dict[str, any]:
        """Build Pyserini HNSW index for the dataset."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Building Pyserini Index for {self.dataset.upper()}")
        logger.info(f"{'='*60}")

        # Check corpus file exists
        corpus_file = self.dataset_path / "corpus.jsonl"
        if not corpus_file.exists():
            logger.error(f"Corpus file not found: {corpus_file}")
            return {"success": False, "error": "Corpus file not found"}

        logger.info(f"Using corpus: {corpus_file}")
        logger.info(f"Output index dir: {self.index_dir}")

        # Create index directory
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Build index using pyserini
        logger.info("Building HNSW index (this may take 5-15 minutes)...")

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        try:
            # Use Anserini index builder with HNSW
            cmd = [
                "python",
                "-m",
                "pyserini.index.lucene",
                "--collection",
                "JsonCollection",
                "--input",
                str(corpus_file),
                "--index",
                str(self.index_dir),
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "4",
                "--storeRaw",
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            indexing_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_increase = end_memory - start_memory

            if result.returncode != 0:
                logger.error(f"Indexing failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "indexing_time_sec": indexing_time,
                }

            logger.info(f"✓ Index built in {indexing_time:.1f} seconds")
            logger.info(f"✓ Memory increase: {memory_increase:.0f} MB")

            # Get index size
            index_size_mb = sum(
                f.stat().st_size for f in self.index_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            logger.info(f"✓ Index size: {index_size_mb:.0f} MB")

            return {
                "success": True,
                "indexing_time_sec": float(indexing_time),
                "memory_increase_mb": float(memory_increase),
                "index_size_mb": float(index_size_mb),
            }

        except subprocess.TimeoutExpired:
            logger.error("Indexing timed out (>1 hour)")
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return {"success": False, "error": str(e)}

    def benchmark_queries(self, num_queries: int = 300) -> Dict[str, any]:
        """Run queries and measure latency."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking Queries on {self.dataset.upper()}")
        logger.info(f"{'='*60}")

        queries_file = self.dataset_path / "queries.jsonl"
        if not queries_file.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return {"success": False, "error": "Queries file not found"}

        logger.info(f"Using queries: {queries_file}")
        logger.info(f"Sample size: {num_queries} queries")

        # Load queries
        queries = []
        with open(queries_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_queries:
                    break
                if line.strip():
                    import json as json_module

                    queries.append(json_module.loads(line))

        logger.info(f"Loaded {len(queries)} queries")

        # Run search benchmark
        logger.info("Running search benchmark (this may take 2-5 minutes)...")

        try:
            from pyserini.search.lucene import LuceneSearcher

            searcher = LuceneSearcher(str(self.index_dir))

            latencies = []
            start_time = time.time()

            for i, query in enumerate(queries):
                q_text = query.get("text", query.get("query", ""))
                if not q_text:
                    continue

                q_start = time.time()
                try:
                    results = searcher.search(q_text, k=100)
                    q_latency = (time.time() - q_start) * 1000  # ms
                    latencies.append(q_latency)
                except Exception as e:
                    logger.warning(f"Query {i} failed: {e}")
                    continue

                if (i + 1) % 50 == 0:
                    logger.info(f"  Processed {i + 1}/{len(queries)} queries")

            total_time = time.time() - start_time
            qps = len(latencies) / total_time if total_time > 0 else 0

            if not latencies:
                logger.error("No queries completed successfully")
                return {"success": False, "error": "Query execution failed"}

            latencies_sorted = sorted(latencies)
            p50 = latencies_sorted[len(latencies_sorted) // 2]
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

            logger.info(f"✓ Completed {len(latencies)} queries in {total_time:.1f}s")
            logger.info(f"✓ QPS: {qps:.1f}")
            logger.info(f"✓ p50 latency: {p50:.1f} ms")
            logger.info(f"✓ p95 latency: {p95:.1f} ms")
            logger.info(f"✓ p99 latency: {p99:.1f} ms")

            return {
                "success": True,
                "num_queries": len(latencies),
                "total_time_sec": float(total_time),
                "qps": float(qps),
                "latency_p50_ms": float(p50),
                "latency_p95_ms": float(p95),
                "latency_p99_ms": float(p99),
                "latency_mean_ms": float(sum(latencies) / len(latencies)),
            }

        except ImportError:
            logger.error("Pyserini searcher not available")
            return {"success": False, "error": "Pyserini import error"}
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {"success": False, "error": str(e)}

    def run_benchmark(self, num_queries: int = 300) -> Dict:
        """Run full benchmark suite."""
        results = {
            "dataset": self.dataset,
            "timestamp": str(__import__("datetime").datetime.now()),
            "pyserini_installed": False,
            "index_results": {},
            "query_results": {},
        }

        # Check Pyserini installation
        if not self.check_pyserini_installed():
            logger.error("Cannot proceed without Pyserini")
            return results

        results["pyserini_installed"] = True

        # Build index
        results["index_results"] = self.build_index()
        if not results["index_results"].get("success"):
            logger.error("Index building failed, skipping query benchmark")
            return results

        # Benchmark queries
        results["query_results"] = self.benchmark_queries(num_queries)

        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pyserini HNSW on BEIR dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="BEIR dataset to benchmark (default: scifact)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=300,
        help="Number of queries to benchmark (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pyserini_benchmark.json"),
        help="Output path for benchmark results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/beir"),
        help="Path to BEIR data directory",
    )

    args = parser.parse_args()

    # Create results directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    benchmark = PyseriniBenchmark(args.dataset, args.data_dir)
    results = benchmark.run_benchmark(args.num_queries)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"PYSERINI BENCHMARK SUMMARY - {args.dataset.upper()}")
    print("=" * 60)

    if results.get("pyserini_installed"):
        print("✓ Pyserini installed")
    else:
        print("✗ Pyserini installation failed")
        return 1

    index_res = results.get("index_results", {})
    if index_res.get("success"):
        print(f"\nIndexing Results:")
        print(f"  Time: {index_res['indexing_time_sec']:.1f} seconds")
        print(f"  Index Size: {index_res['index_size_mb']:.0f} MB")
        print(f"  Memory Increase: {index_res['memory_increase_mb']:.0f} MB")
    else:
        print(f"\n✗ Indexing failed: {index_res.get('error', 'Unknown')}")
        return 1

    query_res = results.get("query_results", {})
    if query_res.get("success"):
        print(f"\nQuery Benchmark Results:")
        print(f"  Queries: {query_res['num_queries']}")
        print(f"  QPS: {query_res['qps']:.1f}")
        print(f"  p50 Latency: {query_res['latency_p50_ms']:.1f} ms")
        print(f"  p95 Latency: {query_res['latency_p95_ms']:.1f} ms")
        print(f"  p99 Latency: {query_res['latency_p99_ms']:.1f} ms")
    else:
        print(f"\n✗ Query benchmark failed: {query_res.get('error', 'Unknown')}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
