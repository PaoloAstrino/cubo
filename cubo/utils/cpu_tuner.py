import logging
import os
from typing import Dict

from cubo.monitoring import metrics
from cubo.utils.hardware import HardwareProfile

logger = logging.getLogger(__name__)


def _calculate_target_threads(cores):
    """Calculate target number of threads based on core count."""
    if cores > 4:
        target_threads = cores - 1
    else:
        target_threads = cores
    return max(1, target_threads)


def _set_environment_variables(env_vars, target_str, dry_run):
    """Set environment variables for threading."""
    changes = {}
    for var in env_vars:
        if var in os.environ:
            logger.debug(f"Skipping {var} (already set to {os.environ[var]})")
            continue

        changes[var] = target_str
        if not dry_run:
            os.environ[var] = target_str
            logger.info(f"Set {var}={target_str}")
    
    return changes


def _set_mkl_threads(profile, target_threads, dry_run):
    """Set MKL threads if MKL backend is available."""
    if dry_run or profile.blas_backend != "mkl":
        return
    
    try:
        import mkl
        mkl.set_num_threads(target_threads)
        logger.info(f"Called mkl.set_num_threads({target_threads})")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to set MKL threads: {e}")


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
    target_threads = _calculate_target_threads(profile.physical_cores)
    target_str = str(target_threads)

    env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]

    changes = _set_environment_variables(env_vars, target_str, dry_run)

    if changes and not dry_run:
        metrics.record("cpu_tuning_applied", 1)

    _set_mkl_threads(profile, target_threads, dry_run)

    return changes
