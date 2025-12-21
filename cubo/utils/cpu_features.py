import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

try:
    import numpy as np
except ImportError:
    np = None


def get_cpu_flags() -> List[str]:
    """
    Detect CPU instruction set flags (AVX, AVX2, AVX512, AMX, etc.).
    Returns a list of lower-case flag strings.
    """
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            # 'flags' is the key on x86, 'features' might be used on ARM/others
            flags = info.get("flags", [])
            if not flags:
                flags = info.get("features", [])
            return [f.lower() for f in flags]
        except Exception:
            return []
    return []


def get_topology() -> Dict[str, int]:
    """
    Get CPU topology: physical cores, logical cores, sockets (if available).
    """
    topology = {
        "physical_cores": 1,
        "logical_cores": 1,
        "sockets": 1,
        "l3_cache_kb": 0
    }

    if psutil:
        try:
            phy = psutil.cpu_count(logical=False)
            log = psutil.cpu_count(logical=True)
            if phy:
                topology["physical_cores"] = phy
            if log:
                topology["logical_cores"] = log
        except Exception:
            pass

    # L3 cache detection is OS-specific and tricky without external tools.
    # cpuinfo sometimes provides 'l3_cache_size' as a string (e.g. "12288 KB").
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            l3 = info.get("l3_cache_size", "")
            if isinstance(l3, str) and "KB" in l3:
                # Parse "12288 KB" -> 12288
                val = l3.replace("KB", "").strip()
                if val.isdigit():
                    topology["l3_cache_kb"] = int(val)
        except Exception:
            pass

    return topology


def detect_blas_backend() -> Tuple[str, Dict[str, str]]:
    """
    Detect the BLAS backend used by NumPy (MKL, OpenBLAS, Accelerate, etc.).
    Returns (backend_name, config_info).
    """
    backend = "unknown"
    config = {}

    if np:
        try:
            # numpy.show_config() prints to stdout, so we inspect internal structures
            # np.__config__.show() is similar.
            # We look at np.__config__.blas_opt_info or similar.
            
            # Modern numpy (1.20+) has np.__config__.get_info or direct dicts
            if hasattr(np.__config__, "get_info"):
                # Try to find specific keys
                if np.__config__.get_info("mkl_info") or np.__config__.get_info("blas_mkl_info"):
                    backend = "mkl"
                elif np.__config__.get_info("openblas_info") or np.__config__.get_info("openblas_lapack_info"):
                    backend = "openblas"
                elif np.__config__.get_info("accelerate_info"):
                    backend = "accelerate"
                elif np.__config__.get_info("blas_opt_info"):
                    # Fallback check
                    info = np.__config__.get_info("blas_opt_info")
                    libs = info.get("libraries", [])
                    if any("mkl" in lib for lib in libs):
                        backend = "mkl"
                    elif any("openblas" in lib for lib in libs):
                        backend = "openblas"
            else:
                # Older numpy or different structure
                # Check loaded libraries via mkl_rt check if possible
                pass

        except Exception:
            pass

    # Secondary check: try importing mkl
    if backend == "unknown":
        try:
            import mkl
            backend = "mkl"
        except ImportError:
            pass

    return backend, config


def detect_allocator() -> str:
    """
    Attempt to detect the memory allocator (libc, jemalloc, tcmalloc).
    This is heuristic-based and checks loaded shared libraries.
    """
    allocator = "libc"  # Default assumption
    
    if psutil and hasattr(psutil.Process, "memory_maps"):
        try:
            proc = psutil.Process()
            maps = proc.memory_maps()
            for m in maps:
                path = m.path.lower()
                if "tcmalloc" in path:
                    return "tcmalloc"
                if "jemalloc" in path:
                    return "jemalloc"
                if "mimalloc" in path:
                    return "mimalloc"
        except Exception:
            pass
            
    return allocator
