"""
Performance utilities for CUBO evaluation.
Tracks latency, memory usage, and hardware stats.
"""

import platform
import time
from typing import Any, Callable, Dict

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def log_hardware_metadata() -> Dict[str, Any]:
    """
    Capture hardware metadata for the current machine.
    Returns:
        Dict with CPU, RAM, and GPU info.
    """
    metadata = {
        "system": platform.system(),
        "release": platform.release(),
        "cpu": {
            "model": platform.processor(),
            "cores_physical": psutil.cpu_count(logical=False) if psutil else 0,
            "cores_logical": psutil.cpu_count(logical=True) if psutil else 0,
        },
        "ram": {
            "total_gb": (
                psutil.virtual_memory().total / (1024**3) if psutil else 0
            ),
        },
        "gpu": {"available": False},
    }

    if torch and torch.cuda.is_available():
        metadata["gpu"] = {
            "available": True,
            "count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory
            / (1024**3),
        }

    return metadata


def sample_memory(sample_duration: float = 0.1, sample_interval: float = 0.01) -> Dict[str, float]:
    """
    Sample memory usage over a duration.
    Args:
        sample_duration: Total time to sample in seconds.
        sample_interval: Interval between samples in seconds.
    Returns:
        Dict with peak memory usage in GB.
    """
    if not psutil:
        return {"ram_peak_gb": 0.0}

    peak_ram = 0
    start = time.time()
    while time.time() - start < sample_duration:
        mem = psutil.virtual_memory().used / (1024**3)
        if mem > peak_ram:
            peak_ram = mem
        time.sleep(sample_interval)

    return {"ram_peak_gb": peak_ram}


def sample_latency(func: Callable, *args, samples: int = 1, **kwargs) -> Dict[str, float]:
    """
    Measure execution latency of a function.
    Args:
        func: Function to measure.
        samples: Number of times to run the function.
    Returns:
        Dict with p50, p95, and p99 latency in ms.
    """
    import numpy as np

    latencies = []
    for _ in range(samples):
        start = time.time()
        func(*args, **kwargs)
        latencies.append((time.time() - start) * 1000)  # Convert to ms

    return {
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else 0,
        "avg_ms": float(np.mean(latencies)) if latencies else 0,
        "min_ms": float(np.min(latencies)) if latencies else 0,
        "max_ms": float(np.max(latencies)) if latencies else 0,
    }
