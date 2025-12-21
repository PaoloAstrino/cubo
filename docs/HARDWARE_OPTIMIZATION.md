# Hardware Optimization in CUBO

CUBO includes advanced hardware detection and optimization features to ensure maximum performance on various hardware configurations, from laptops to high-end servers.

## Hardware Detection

CUBO automatically detects your system's hardware capabilities on startup. This includes:

- **CPU Features**: Detection of AVX, AVX2, AVX512 instruction sets.
- **BLAS Backend**: Identification of the underlying linear algebra library (MKL, OpenBLAS, etc.).
- **Topology**: Detection of physical vs. logical cores and socket count.
- **Memory**: Total available RAM.

This information is used to create a `HardwareProfile` that guides optimization decisions.

## CPU Tuning

For CPU-bound workloads (like embedding generation and retrieval on CPU), CUBO offers an opt-in auto-tuner.

### How it works

The tuner calculates the optimal number of threads for linear algebra operations based on your physical core count. It avoids oversubscription (using logical cores/hyperthreading for compute-heavy tasks) which can degrade performance.

It sets the following environment variables if they are not already set:
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `NUMEXPR_NUM_THREADS`

### Usage

To enable CPU tuning, you can use the `auto_tune_cpu` utility or configure it in your startup script.

```python
from cubo.utils.cpu_tuner import auto_tune_cpu
from cubo.utils.hardware import detect_hardware

profile = detect_hardware()
auto_tune_cpu(profile)
```

## Hardware-Aware Model Loading

The embedding model loader uses the hardware profile to make intelligent decisions:

- **Quantization**: If your CPU supports AVX2 or AVX512, CUBO can prefer quantized models (int8) which offer significant speedups with minimal accuracy loss.
- **Backend Selection**: (Future) Automatically selecting ONNX or OpenVINO backends based on available hardware.

### Configuration

You can control this behavior via `config.json`:

```json
{
  "embeddings": {
    "prefer_quantized_cpu": "auto" // Options: "auto", "always", "never"
  }
}
```

- `auto`: Use quantized models if AVX2/AVX512 is detected.
- `always`: Force usage of quantized models (if available).
- `never`: Always use standard float32 models.

## Troubleshooting

If you experience performance issues, check the logs for "Hardware Profile" to see what CUBO detected.

```
INFO: Hardware Profile: CPU=Intel Core i7-10750H, Cores=6/12, Flags=[AVX2, FMA3], BLAS=mkl
```
