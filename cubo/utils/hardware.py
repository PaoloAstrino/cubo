from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch

from cubo.utils import cpu_features


@dataclass
class HardwareProfile:
    device: Literal["cuda", "mps", "cpu"]
    n_gpu_layers: int
    vram_gb: float

    # Extended CPU/System info
    physical_cores: int = 1
    logical_cores: int = 1
    total_ram_gb: float = 0.0
    cpu_flags: List[str] = field(default_factory=list)
    blas_backend: str = "unknown"
    allocator: str = "unknown"


def detect_hardware() -> HardwareProfile:
    """
    Detects available hardware acceleration (CUDA, MPS) and system capabilities.

    Returns:
        HardwareProfile with device type, VRAM, CPU topology, and feature flags.
    """
    device = "cpu"
    n_gpu_layers = 0
    vram_gb = 0.0

    # 1. GPU Detection
    try:
        if torch.cuda.is_available():
            device = "cuda"
            n_gpu_layers = -1
            try:
                # Get VRAM of the first device
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                pass
        elif torch.backends.mps.is_available():
            device = "mps"
            n_gpu_layers = -1
            # MPS VRAM is shared with system memory
    except Exception:
        # Fallback to CPU if torch detection fails
        pass

    # 2. CPU & System Detection
    topology = cpu_features.get_topology()
    flags = cpu_features.get_cpu_flags()
    blas, _ = cpu_features.detect_blas_backend()
    allocator = cpu_features.detect_allocator()

    # Get RAM
    total_ram = 0.0
    try:
        import psutil

        total_ram = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    return HardwareProfile(
        device=device,
        n_gpu_layers=n_gpu_layers,
        vram_gb=vram_gb,
        physical_cores=topology.get("physical_cores", 1),
        logical_cores=topology.get("logical_cores", 1),
        total_ram_gb=total_ram,
        cpu_flags=flags,
        blas_backend=blas,
        allocator=allocator,
    )
