import torch
from dataclasses import dataclass
from typing import Literal

@dataclass
class HardwareProfile:
    device: Literal["cuda", "mps", "cpu"]
    n_gpu_layers: int
    vram_gb: float

def detect_hardware() -> HardwareProfile:
    """
    Detects available hardware acceleration (CUDA, MPS) and returns a profile.
    
    Returns:
        HardwareProfile with device type, recommended gpu layers (-1 for all), and VRAM.
    """
    device = "cpu"
    n_gpu_layers = 0
    vram_gb = 0.0

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
            # MPS VRAM is shared with system memory, hard to pin down exact "VRAM"
            # but we can treat it as having acceleration.
    except Exception:
        # Fallback to CPU if torch detection fails
        pass

    return HardwareProfile(device=device, n_gpu_layers=n_gpu_layers, vram_gb=vram_gb)
