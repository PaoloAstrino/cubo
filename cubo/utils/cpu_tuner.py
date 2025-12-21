import os
import logging
from typing import Dict, Optional

from cubo.utils.hardware import HardwareProfile
from cubo.monitoring import metrics

logger = logging.getLogger(__name__)


def auto_tune_cpu(profile: HardwareProfile, dry_run: bool = False) -> Dict[str, str]:
    """
    Automatically tune CPU environment variables for optimal performance based on hardware profile.
    
    This function sets threading environment variables (OMP_NUM_THREADS, etc.) based on
    physical core counts. It respects existing environment variables and will not overwrite them.
    
    Args:
        profile: The detected hardware profile.
        dry_run: If True, only return what would be changed without applying it.
        
    Returns:
        Dictionary of environment variables that were (or would be) set.
    """
    changes = {}
    
    # Heuristic: Use physical cores for compute-heavy tasks
    # But leave some headroom for system if we have many cores
    # If we have few cores (<=4), use all of them.
    # If we have many, maybe reserve 1 core for system/IO.
    
    cores = profile.physical_cores
    if cores > 4:
        # Reserve 1 core for system/IO if we have plenty
        target_threads = cores - 1
    else:
        target_threads = cores
        
    # Ensure at least 1 thread
    target_threads = max(1, target_threads)
    target_str = str(target_threads)
    
    # List of env vars to tune
    # We prioritize MKL and OpenBLAS specific ones, then generic OMP
    env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    
    for var in env_vars:
        if var in os.environ:
            logger.debug(f"Skipping {var} (already set to {os.environ[var]})")
            continue
            
        changes[var] = target_str
        
        if not dry_run:
            os.environ[var] = target_str
            logger.info(f"Set {var}={target_str}")
            
    if changes and not dry_run:
        metrics.record("cpu_tuning_applied", 1)
            
    # Attempt to set runtime MKL threads if available and not dry run
    # This is useful if MKL is already loaded and might have initialized with a different default
    if not dry_run and profile.blas_backend == "mkl":
        try:
            import mkl
            mkl.set_num_threads(target_threads)
            logger.info(f"Called mkl.set_num_threads({target_threads})")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to set MKL threads: {e}")
            
    return changes
