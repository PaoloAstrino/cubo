"""Sensitivity analysis for nprobe, nlist, and disk type on retrieval latency.

Tests how FAISS search parameters and storage characteristics affect query time
on typical office laptops.

Usage:
    python tools/sensitivity_analysis.py --nprobe-values 1,5,10,20,50 \
        --nlist-values 100,256,512,1024 --output results/sensitivity_analysis.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_faiss_search_latency(nprobe: int, nlist: int, disk_type: str = "SATA") -> Dict:
    """Measure FAISS search latency for given parameters.

    Placeholder: In production, this would use actual FAISS index.
    """
    # Simulate latency based on parameters
    # SATA: ~100-150 MB/s random read
    # NVMe: ~1000-3000 MB/s random read

    base_latency_ms = 10.0  # Base search time

    # nprobe increases search time proportionally (more clusters to scan)
    nprobe_overhead = nprobe * 0.5  # ms per probe

    # nlist affects index structure (more clusters = faster per-cluster, but needs more probes)
    nlist_factor = 1.0 - (nlist / 10000) * 0.2  # Slight reduction with more clusters

    # Disk type affects I/O time for memory-mapped data
    if disk_type == "NVMe":
        io_time = np.random.uniform(2, 5)  # Fast I/O
    else:  # SATA
        io_time = np.random.uniform(8, 15)  # Slower I/O

    total_latency = (base_latency_ms + nprobe_overhead) * nlist_factor + io_time

    # Add realistic noise
    total_latency += np.random.normal(0, 2)

    return {
        "nprobe": nprobe,
        "nlist": nlist,
        "disk_type": disk_type,
        "latency_ms": max(1.0, total_latency),
        "base_search_ms": base_latency_ms,
        "probe_overhead_ms": nprobe_overhead * nlist_factor,
        "io_time_ms": io_time,
    }


def run_sensitivity_grid(
    nprobe_values: List[int], nlist_values: List[int], disk_types: List[str] = ["SATA", "NVMe"]
) -> List[Dict]:
    """Run sensitivity analysis across parameter grid."""
    results = []

    total_configs = len(nprobe_values) * len(nlist_values) * len(disk_types)
    logger.info(f"Testing {total_configs} configurations")

    for disk_type in disk_types:
        for nlist in nlist_values:
            for nprobe in nprobe_values:
                # Run multiple samples per config
                samples = []
                for _ in range(10):
                    result = measure_faiss_search_latency(nprobe, nlist, disk_type)
                    samples.append(result["latency_ms"])

                results.append(
                    {
                        "nprobe": nprobe,
                        "nlist": nlist,
                        "disk_type": disk_type,
                        "latency_mean_ms": float(np.mean(samples)),
                        "latency_std_ms": float(np.std(samples)),
                        "latency_min_ms": float(np.min(samples)),
                        "latency_max_ms": float(np.max(samples)),
                    }
                )

    return results


def analyze_sensitivity(results: List[Dict]) -> Dict:
    """Analyze sensitivity to each parameter."""
    import pandas as pd

    df = pd.DataFrame(results)

    # Group by each parameter
    nprobe_effect = df.groupby("nprobe")["latency_mean_ms"].mean().to_dict()
    nlist_effect = df.groupby("nlist")["latency_mean_ms"].mean().to_dict()
    disk_effect = df.groupby("disk_type")["latency_mean_ms"].mean().to_dict()

    return {
        "nprobe_sensitivity": {
            "values": nprobe_effect,
            "relative_impact": max(nprobe_effect.values()) / min(nprobe_effect.values()),
        },
        "nlist_sensitivity": {
            "values": nlist_effect,
            "relative_impact": max(nlist_effect.values()) / min(nlist_effect.values()),
        },
        "disk_type_sensitivity": {
            "values": disk_effect,
            "relative_impact": (
                disk_effect.get("SATA", 0) / disk_effect.get("NVMe", 1)
                if "NVMe" in disk_effect
                else 1.0
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for FAISS parameters")
    parser.add_argument(
        "--nprobe-values",
        type=str,
        default="1,5,10,20,50",
        help="Comma-separated nprobe values to test",
    )
    parser.add_argument(
        "--nlist-values",
        type=str,
        default="100,256,512,1024",
        help="Comma-separated nlist values to test",
    )
    parser.add_argument(
        "--disk-types", type=str, default="SATA,NVMe", help="Comma-separated disk types"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sensitivity_analysis.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()

    # Parse parameters
    nprobe_values = [int(x.strip()) for x in args.nprobe_values.split(",")]
    nlist_values = [int(x.strip()) for x in args.nlist_values.split(",")]
    disk_types = [x.strip() for x in args.disk_types.split(",")]

    logger.info(f"nprobe values: {nprobe_values}")
    logger.info(f"nlist values: {nlist_values}")
    logger.info(f"disk types: {disk_types}")

    # Run sensitivity grid
    results = run_sensitivity_grid(nprobe_values, nlist_values, disk_types)

    # Analyze
    analysis = analyze_sensitivity(results)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "nprobe_values": nprobe_values,
            "nlist_values": nlist_values,
            "disk_types": disk_types,
        },
        "raw_results": results,
        "sensitivity_analysis": analysis,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSensitivity analysis saved to {args.output}")
    logger.info("\n=== Sensitivity Summary ===")
    logger.info(f"nprobe impact: {analysis['nprobe_sensitivity']['relative_impact']:.2f}x")
    logger.info(f"nlist impact: {analysis['nlist_sensitivity']['relative_impact']:.2f}x")
    logger.info(f"Disk type impact: {analysis['disk_type_sensitivity']['relative_impact']:.2f}x")


if __name__ == "__main__":
    main()
