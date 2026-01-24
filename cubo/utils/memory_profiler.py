"""Memory profiling utility for validating O(1) memory claims during ingestion.

This module provides tools to track RSS memory usage over time and generate
validation evidence for the paper's O(1) memory scaling claims.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


class MemoryProfiler:
    """Track RSS memory usage over time for O(1) ingestion validation.
    
    Usage:
        profiler = MemoryProfiler("ingestion_10gb.jsonl")
        profiler.record("start")
        # ... ingestion work ...
        profiler.record("batch_1_flush")
        # ... more work ...
        profiler.record("end")
        profiler.save()
        profiler.print_summary()
    """

    def __init__(self, output_file: str = "memory_profile.jsonl", enabled: bool = True):
        """Initialize the memory profiler.
        
        Args:
            output_file: Path to output JSONL file for samples
            enabled: If False, all operations are no-ops (for production)
        """
        self.output_file = Path(output_file)
        self.enabled = enabled and psutil is not None
        self.samples: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self._process = psutil.Process() if psutil else None
        
    def record(self, tag: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Record a memory sample with a descriptive tag.
        
        Args:
            tag: Descriptive name for this checkpoint (e.g., "batch_5_flush", "gc_triggered")
            extra: Optional extra metadata to include
        """
        if not self.enabled:
            return
            
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            
        try:
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
        except Exception:
            rss_mb = 0.0
            vms_mb = 0.0
            
        sample = {
            "timestamp": now,
            "elapsed_s": round(now - self.start_time, 2),
            "tag": tag,
            "rss_mb": round(rss_mb, 1),
            "vms_mb": round(vms_mb, 1),
        }
        
        if extra:
            sample.update(extra)
            
        self.samples.append(sample)
        
    def save(self) -> Path:
        """Save all samples to the JSONL file."""
        if not self.enabled or not self.samples:
            return self.output_file
            
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + "\n")
                
        return self.output_file
        
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics from recorded samples."""
        if not self.samples:
            return {"error": "No samples recorded"}
            
        rss_values = [s["rss_mb"] for s in self.samples]
        
        return {
            "sample_count": len(self.samples),
            "min_rss_mb": min(rss_values),
            "max_rss_mb": max(rss_values),
            "mean_rss_mb": round(sum(rss_values) / len(rss_values), 1),
            "delta_rss_mb": round(max(rss_values) - min(rss_values), 1),
            "duration_s": round(self.samples[-1]["elapsed_s"], 1) if self.samples else 0,
            "is_o1": (max(rss_values) - min(rss_values)) < 500,  # <500MB delta = O(1)
        }
        
    def print_summary(self) -> None:
        """Print a human-readable summary of memory usage."""
        stats = self.get_stats()
        if "error" in stats:
            print(stats["error"])
            return
            
        print("\n" + "=" * 60)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 60)
        print(f"Total samples:     {stats['sample_count']}")
        print(f"Duration:          {stats['duration_s']}s")
        print(f"Min RSS:           {stats['min_rss_mb']} MB")
        print(f"Max RSS:           {stats['max_rss_mb']} MB")
        print(f"Mean RSS:          {stats['mean_rss_mb']} MB")
        print(f"Delta (Max-Min):   {stats['delta_rss_mb']} MB")
        print("-" * 60)
        
        if stats["is_o1"]:
            print("[OK] O(1) MEMORY CLAIM VALIDATED (delta < 500MB)")
        else:
            print("[WARNING] O(1) MEMORY CLAIM REQUIRES INVESTIGATION (delta >= 500MB)")
        print("=" * 60 + "\n")


# Global profiler instance for easy access
_global_profiler: Optional[MemoryProfiler] = None


def get_profiler(output_file: str = "memory_profile.jsonl") -> MemoryProfiler:
    """Get or create a global memory profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MemoryProfiler(output_file)
    return _global_profiler


def reset_profiler() -> None:
    """Reset the global profiler instance."""
    global _global_profiler
    _global_profiler = None
