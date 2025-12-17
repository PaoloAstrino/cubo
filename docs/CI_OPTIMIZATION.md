# CI/CD Optimization Guide

## Problem
CI/CD pipelines were failing with `[Errno 28] No space left on device` when installing dependencies.

## Root Cause
The full `requirements.txt` includes heavy ML packages:
- `torch` (~2GB)
- `torchvision` (~500MB)
- `transformers` (~500MB)
- `sentence-transformers`
- `easyocr` (~300MB)
- `pymupdf` (~200MB)

Total: **~5GB** of dependencies, exceeding GitHub Actions runner disk space limits.

## Solution

### 1. Use `requirements-ci.txt` (Minimal Install)
For CI/CD pipelines, use the lightweight requirements file:
```bash
pip install --no-cache-dir -r requirements-ci.txt
```

This reduces installation size from **~5GB to ~500MB** (10x smaller).

### 2. Add `--no-cache-dir` Flag
Always use `--no-cache-dir` in CI to prevent pip from caching downloaded packages:
```bash
pip install --no-cache-dir -r requirements-ci.txt
```

### 3. Free Disk Space (GitHub Actions)
Add disk cleanup step before installation:
```yaml
- name: Free Disk Space (Ubuntu)
  if: runner.os == 'Linux'
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/ghc
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo docker image prune --all --force
    df -h
```

This frees **~30GB** of space on GitHub Actions runners.

## Requirements Files Overview

| File | Size | Use Case |
|------|------|----------|
| `requirements-minimal.txt` | ~200MB | Ollama-only mode (no PyTorch) |
| `requirements-ci.txt` | ~500MB | **CI/CD pipelines** (testing/linting) |
| `requirements-core.txt` | ~1.5GB | Laptop mode (includes sentence-transformers) |
| `requirements.txt` | ~5GB | Full production (all features) |

## Testing Locally

To simulate CI environment:
```bash
# Create fresh venv
python -m venv .venv-ci
source .venv-ci/bin/activate  # On Windows: .venv-ci\Scripts\activate

# Install CI dependencies
pip install --no-cache-dir -r requirements-ci.txt
pip install --no-cache-dir -r requirements-dev.txt
pip install --no-cache-dir -e .

# Run tests
pytest tests/unit -v
```

## What's Excluded from CI

Heavy packages excluded from `requirements-ci.txt`:
- ❌ `torch`, `torchvision` - Use Ollama embeddings instead
- ❌ `transformers`, `sentence-transformers` - Not needed for testing
- ❌ `easyocr` - OCR not required for unit tests
- ❌ `pymupdf` - Use `pypdf` (lighter alternative)
- ❌ `ragas`, `langchain` - Benchmark-only dependencies

## CI Workflow Changes

All workflows now use:
1. ✅ `requirements-ci.txt` instead of `requirements.txt`
2. ✅ `--no-cache-dir` flag on all pip installs
3. ✅ Disk space cleanup step for Linux runners
4. ✅ Minimal pyproject.toml extras: `.[dev]` only

## Migration Checklist

- [x] Create `requirements-ci.txt`
- [x] Update `.github/workflows/reusable-quality.yml`
- [x] Update `.github/workflows/pr-checks.yml`
- [x] Update `.github/workflows/main-validation.yml`
- [x] Add disk cleanup steps
- [x] Document changes

## Results

**Before:**
- Install size: ~5GB
- Pipeline status: ❌ FAILED (disk full)

**After:**
- Install size: ~500MB
- Free space: +30GB (from cleanup)
- Pipeline status: ✅ PASSING

## References

- [GitHub Actions Runner Images](https://github.com/actions/runner-images)
- [Pip No Cache Dir](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-cache-dir)
- [Free Disk Space Action](https://github.com/jlumbroso/free-disk-space)
