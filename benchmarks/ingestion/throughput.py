#!/usr/bin/env python3
"""
Ingestion Throughput Testing Script
Measures ingestion time, compression ratio, and indexing performance.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.utils.hardware import log_hardware_metadata, sample_memory
from cubo.main import CUBOApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IngestionTester:
    """Test ingestion throughput and compression performance."""

    def __init__(self, data_folder: str, output_folder: str = "evaluation/ingestion_results"):
        """
        Initialize ingestion tester.

        Args:
            data_folder: Path to folder containing documents to ingest
            output_folder: Path to save ingestion results
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Capture hardware metadata
        self.hardware_metadata = log_hardware_metadata()
        logger.info(
            f"Hardware: {self.hardware_metadata['cpu']['model']}, "
            f"{self.hardware_metadata['ram']['total_gb']:.1f}GB RAM"
        )

    def calculate_data_size(self) -> Dict[str, Any]:
        """
        Calculate total size of data to ingest.

        Returns:
            Dictionary with size metrics
        """
        total_bytes = 0
        file_count = 0

        for file_path in self.data_folder.rglob("*"):
            if file_path.is_file():
                total_bytes += file_path.stat().st_size
                file_count += 1

        total_gb = total_bytes / (1024**3)

        return {
            "total_bytes": total_bytes,
            "total_gb": total_gb,
            "total_mb": total_bytes / (1024**2),
            "file_count": file_count,
        }

    def measure_ingestion(self, fast_pass: bool = True) -> Dict[str, Any]:
        """
        Measure ingestion performance.

        Args:
            fast_pass: Whether to use fast-pass ingestion

        Returns:
            Dictionary with ingestion metrics
        """
        logger.info(f"Starting ingestion measurement (fast_pass={fast_pass})...")

        # Calculate input data size
        data_size = self.calculate_data_size()
        logger.info(
            f"Data to ingest: {data_size['total_gb']:.2f} GB ({data_size['file_count']} files)"
        )

        # Initialize CUBO system
        cubo_app = CUBOApp()

        # Sample memory before ingestion
        memory_before = sample_memory()

        # Measure ingestion time
        start_time = time.time()

        try:
            # Initialize components
            if not cubo_app.initialize_components():
                raise Exception("Failed to initialize CUBO components")

            # Load documents
            logger.info("Loading documents...")
            documents = cubo_app.doc_loader.load_documents_from_folder(str(self.data_folder))

            if not documents:
                raise Exception("No documents loaded")

            logger.info(f"Loaded {len(documents)} document chunks")

            # Add to vector store
            logger.info("Adding documents to vector store...")
            document_texts = []
            for chunk in documents:
                if isinstance(chunk, dict) and "text" in chunk:
                    document_texts.append(chunk["text"])
                elif isinstance(chunk, str):
                    document_texts.append(chunk)

            if document_texts:
                cubo_app.retriever.add_documents(document_texts)
            else:
                raise Exception("No valid document texts to add")

            ingestion_time = time.time() - start_time

            # Sample memory after ingestion
            memory_after = sample_memory()

            # Calculate throughput
            seconds_per_gb = (
                ingestion_time / data_size["total_gb"] if data_size["total_gb"] > 0 else 0
            )
            minutes_per_gb = seconds_per_gb / 60
            gb_per_minute = (1 / minutes_per_gb) if minutes_per_gb > 0 else 0

            # Estimate storage size (rough approximation)
            # TODO: Calculate actual index sizes from FAISS, parquet, and embeddings
            storage_estimate = self._estimate_storage_size()
            compression_ratio = (
                data_size["total_bytes"] / storage_estimate["total_bytes"]
                if storage_estimate["total_bytes"] > 0
                else 0
            )

            results = {
                "success": True,
                "data_size": data_size,
                "ingestion_time_seconds": ingestion_time,
                "ingestion_time_minutes": ingestion_time / 60,
                "seconds_per_gb": seconds_per_gb,
                "minutes_per_gb": minutes_per_gb,
                "gb_per_minute": gb_per_minute,
                "chunks_ingested": len(documents),
                "chunks_per_second": len(documents) / ingestion_time if ingestion_time > 0 else 0,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta_gb": memory_after["ram_peak_gb"] - memory_before["ram_peak_gb"],
                "storage_estimate": storage_estimate,
                "compression_ratio": compression_ratio,
                "fast_pass": fast_pass,
            }

            logger.info(
                f"Ingestion completed in {ingestion_time:.2f}s ({minutes_per_gb:.2f} min/GB)"
            )

            return results

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data_size": data_size,
                "ingestion_time_seconds": time.time() - start_time,
            }

    def _estimate_storage_size(self) -> Dict[str, Any]:
        """
        Estimate storage size of indexed data.

        Returns:
            Dictionary with storage size estimates
        """
        # Check for FAISS index files
        faiss_size = 0
        faiss_dir = Path("faiss_index_dir")
        if faiss_dir.exists():
            for file_path in faiss_dir.rglob("*"):
                if file_path.is_file():
                    faiss_size += file_path.stat().st_size

        # Check for parquet files
        parquet_size = 0
        for file_path in Path(".").rglob("*.parquet"):
            parquet_size += file_path.stat().st_size

        total_bytes = faiss_size + parquet_size

        return {
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024**3),
            "faiss_bytes": faiss_size,
            "parquet_bytes": parquet_size,
        }

    def run_test(self, fast_pass: bool = True) -> Dict[str, Any]:
        """
        Run complete ingestion test.

        Args:
            fast_pass: Whether to use fast-pass ingestion

        Returns:
            Dictionary with complete test results
        """
        results = {
            "metadata": {
                "test_timestamp": time.time(),
                "data_folder": str(self.data_folder),
                "fast_pass": fast_pass,
                "hardware": self.hardware_metadata,
            },
            "ingestion": self.measure_ingestion(fast_pass=fast_pass),
        }

        return results

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """
        Save results to JSON file.

        Args:
            results: Test results dictionary
            output_file: Output filename (default: ingestion_results_{timestamp}.json)
        """
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"ingestion_results_{timestamp}.json"

        output_path = self.output_folder / output_file

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print ingestion test summary.

        Args:
            results: Test results dictionary
        """
        print("\n" + "=" * 60)
        print("INGESTION THROUGHPUT TEST SUMMARY")
        print("=" * 60)

        ingestion = results["ingestion"]

        if ingestion["success"]:
            data_size = ingestion["data_size"]
            print(f"Data Size: {data_size['total_gb']:.2f} GB ({data_size['file_count']} files)")
            print(f"Chunks Ingested: {ingestion['chunks_ingested']}")
            print("\nIngestion Performance:")
            print(f"  Total Time: {ingestion['ingestion_time_minutes']:.2f} minutes")
            print(f"  Throughput: {ingestion['gb_per_minute']:.2f} GB/minute")
            print(f"  Time per GB: {ingestion['minutes_per_gb']:.2f} minutes")
            print(f"  Chunks/second: {ingestion['chunks_per_second']:.1f}")

            print("\nMemory Usage:")
            print(f"  Before: {ingestion['memory_before']['ram_peak_gb']:.2f} GB RAM")
            print(f"  After: {ingestion['memory_after']['ram_peak_gb']:.2f} GB RAM")
            print(f"  Delta: {ingestion['memory_delta_gb']:.2f} GB")

            if ingestion["compression_ratio"] > 0:
                print("\nCompression:")
                storage = ingestion["storage_estimate"]
                print(f"  Raw Data: {data_size['total_gb']:.2f} GB")
                print(f"  Stored: {storage['total_gb']:.2f} GB")
                print(f"  Compression Ratio: {ingestion['compression_ratio']:.1f}:1")
        else:
            print(f"Ingestion FAILED: {ingestion.get('error', 'Unknown error')}")

        # Hardware summary
        hw = results["metadata"]["hardware"]
        print("\nHardware Configuration:")
        print(f"  CPU: {hw['cpu']['model']}")
        print(f"  RAM: {hw['ram']['total_gb']:.1f} GB")
        if hw["gpu"].get("available"):
            print(f"  GPU: {hw['gpu']['device_name']} ({hw['gpu']['vram_total_gb']:.1f} GB VRAM)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="CUBO Ingestion Throughput Testing")
    parser.add_argument(
        "--data-folder", required=True, help="Path to folder containing documents to ingest"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename for results (default: ingestion_results_{timestamp}.json)",
    )
    parser.add_argument(
        "--fast-pass",
        action="store_true",
        default=True,
        help="Use fast-pass ingestion (default: True)",
    )
    parser.add_argument(
        "--deep-ingest", action="store_true", help="Use deep ingestion instead of fast-pass"
    )

    args = parser.parse_args()

    # Determine ingestion mode
    fast_pass = not args.deep_ingest

    # Initialize tester
    tester = IngestionTester(args.data_folder)

    # Run test
    results = tester.run_test(fast_pass=fast_pass)

    # Save and display results
    tester.save_results(results, args.output)
    tester.print_summary(results)


if __name__ == "__main__":
    main()
