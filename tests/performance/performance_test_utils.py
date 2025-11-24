"""
Performance testing utilities with statistical rigor.

Provides helpers for:
- Memory measurement with tracemalloc
- Statistical timing with warmup and outlier detection
- Throughput calculation
- Regression analysis for complexity testing
"""

import gc
import time
import timeit
import tracemalloc
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import psutil


class MemoryProfiler:
    """Measure memory usage with tracemalloc."""
    
    @staticmethod
    def measure_peak_memory(operation: Callable, *args, **kwargs) -> Dict[str, float]:
        """
        Measure peak memory usage of an operation using tracemalloc.
        
        Args:
            operation: Function to measure
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Dictionary with current_mb, peak_mb
        """
        gc.collect()
        tracemalloc.start()
        
        try:
            result = operation(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'result': result
        }
    
    @staticmethod
    def measure_memory_stats(operation: Callable, n_samples: int = 10, 
                           *args, **kwargs) -> Dict[str, float]:
        """
        Measure memory with multiple samples and return statistics.
        
        Args:
            operation: Function to measure
            n_samples: Number of measurement samples
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Dictionary with mean_mb, std_mb, min_mb, max_mb, median_mb
        """
        peak_samples = []
        
        for _ in range(n_samples):
            gc.collect()
            tracemalloc.start()
            
            try:
                operation(*args, **kwargs)
                _, peak = tracemalloc.get_traced_memory()
                peak_samples.append(peak / 1024 / 1024)
            finally:
                tracemalloc.stop()
        
        return {
            'mean_mb': np.mean(peak_samples),
            'std_mb': np.std(peak_samples),
            'min_mb': np.min(peak_samples),
            'max_mb': np.max(peak_samples),
            'median_mb': np.median(peak_samples),
            'samples': peak_samples
        }


class LatencyProfiler:
    """Measure latency with statistical rigor."""
    
    @staticmethod
    def measure_with_warmup(operation: Callable, warmup_runs: int = 3,
                          n_samples: int = 10, *args, **kwargs) -> Dict[str, float]:
        """
        Measure operation latency with warmup and statistical analysis.
        
        Args:
            operation: Function to measure
            warmup_runs: Number of warmup executions (discarded)
            n_samples: Number of measurement samples
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Dictionary with mean_ms, median_ms, p50_ms, p95_ms, p99_ms, std_ms
        """
        # Warmup phase
        for _ in range(warmup_runs):
            operation(*args, **kwargs)
        
        # Measurement phase
        def wrapped_operation():
            operation(*args, **kwargs)
        
        times = timeit.repeat(wrapped_operation, repeat=n_samples, number=1)
        
        # Remove outliers (top 10%)
        times_sorted = sorted(times)
        times_clean = times_sorted[:int(len(times_sorted) * 0.9)]
        
        # Convert to milliseconds
        times_ms = [t * 1000 for t in times_clean]
        all_times_ms = [t * 1000 for t in times]
        
        return {
            'mean_ms': np.mean(times_ms),
            'median_ms': np.median(times_ms),
            'std_ms': np.std(times_ms),
            'min_ms': np.min(all_times_ms),
            'max_ms': np.max(all_times_ms),
            'p50_ms': np.percentile(all_times_ms, 50),
            'p95_ms': np.percentile(all_times_ms, 95),
            'p99_ms': np.percentile(all_times_ms, 99),
            'samples_ms': all_times_ms
        }


class ThroughputProfiler:
    """Measure throughput (items/second)."""
    
    @staticmethod
    def measure_throughput(operation: Callable, n_items: int,
                          warmup: bool = True, *args, **kwargs) -> Dict[str, float]:
        """
        Measure throughput in items per second.
        
        Args:
            operation: Function to measure
            n_items: Number of items processed by operation
            warmup: Whether to do warmup run
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Dictionary with items_per_sec, total_time_sec, avg_time_per_item_ms
        """
        if warmup:
            operation(*args, **kwargs)
        
        start = time.perf_counter()
        operation(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        throughput = n_items / elapsed if elapsed > 0 else 0
        avg_time_per_item = (elapsed / n_items * 1000) if n_items > 0 else 0
        
        return {
            'items_per_sec': throughput,
            'total_time_sec': elapsed,
            'avg_time_per_item_ms': avg_time_per_item
        }


class ComplexityAnalyzer:
    """Analyze algorithmic complexity through regression."""
    
    @staticmethod
    def analyze_complexity(operation: Callable, sizes: List[int],
                         expected_complexity: str = 'linear',
                         *args, **kwargs) -> Dict[str, Any]:
        """
        Analyze complexity by fitting to different growth models.
        
        Args:
            operation: Function that takes size as first argument
            sizes: List of input sizes to test
            expected_complexity: 'linear', 'nlogn', or 'quadratic'
            *args, **kwargs: Additional arguments to operation
            
        Returns:
            Dictionary with complexity analysis results including r_squared
        """
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score
        
        times = []
        for size in sizes:
            # Warmup
            operation(size, *args, **kwargs)
            
            # Measure
            start = time.perf_counter()
            operation(size, *args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Define complexity models
        def linear(n, a, b):
            return a * n + b
        
        def nlogn(n, a, b):
            return a * n * np.log(n + 1) + b  # +1 to avoid log(0)
        
        def quadratic(n, a, b):
            return a * n**2 + b
        
        models = {
            'linear': linear,
            'nlogn': nlogn,
            'quadratic': quadratic
        }
        
        # Fit all models
        results = {}
        for name, model in models.items():
            try:
                params, _ = curve_fit(model, sizes, times, maxfev=10000)
                predictions = [model(s, *params) for s in sizes]
                r2 = r2_score(times, predictions)
                results[name] = {
                    'r_squared': r2,
                    'params': params.tolist()
                }
            except Exception as e:
                results[name] = {
                    'r_squared': -1,
                    'error': str(e)
                }
        
        # Determine best fit
        best_fit = max(results.items(), key=lambda x: x[1].get('r_squared', -1))
        
        return {
            'sizes': sizes,
            'times_sec': times,
            'fits': results,
            'best_fit': best_fit[0],
            'best_r_squared': best_fit[1].get('r_squared', -1),
            'matches_expected': best_fit[0] == expected_complexity
        }


class ResourceMonitor:
    """Monitor system resources."""
    
    @staticmethod
    def check_file_handles(operation: Callable, *args, **kwargs) -> Dict[str, int]:
        """
        Check if file handles are properly cleaned up.
        
        Returns:
            Dictionary with handles_before, handles_after, handles_leaked
        """
        proc = psutil.Process()
        
        # Force GC before counting
        gc.collect()
        handles_before = len(proc.open_files())
        
        operation(*args, **kwargs)
        
        # Force GC after operation
        gc.collect()
        time.sleep(0.1)  # Give OS time to close handles
        handles_after = len(proc.open_files())
        
        return {
            'handles_before': handles_before,
            'handles_after': handles_after,
            'handles_leaked': max(0, handles_after - handles_before)
        }
    
    @staticmethod
    def measure_memory_growth(operation: Callable, iterations: int = 10,
                            *args, **kwargs) -> Dict[str, float]:
        """
        Measure if memory grows over repeated operations (leak detection).
        
        Returns:
            Dictionary with initial_mb, final_mb, growth_mb, growth_per_iter_mb
        """
        proc = psutil.Process()
        
        gc.collect()
        initial_mem = proc.memory_info().rss / 1024 / 1024
        
        for _ in range(iterations):
            operation(*args, **kwargs)
        
        gc.collect()
        time.sleep(0.1)
        final_mem = proc.memory_info().rss / 1024 / 1024
        
        growth = final_mem - initial_mem
        growth_per_iter = growth / iterations if iterations > 0 else 0
        
        return {
            'initial_mb': initial_mem,
            'final_mb': final_mem,
            'growth_mb': growth,
            'growth_per_iter_mb': growth_per_iter,
            'has_leak': growth_per_iter > 1.0  # >1MB per iteration suggests leak
        }
