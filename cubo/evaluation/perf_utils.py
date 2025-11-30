"""
Performance utilities for CUBO evaluation.
Re-exports from benchmarks.utils.hardware for backward compatibility.
"""

from benchmarks.utils.hardware import (
    log_hardware_metadata,
    sample_latency,
    sample_memory,
)

__all__ = [
    "log_hardware_metadata",
    "sample_latency", 
    "sample_memory",
]
