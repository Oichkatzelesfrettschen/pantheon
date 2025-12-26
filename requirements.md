# Project Requirements

To ensure a reproducible build, the following dependencies are required. These versions are pinned based on the current validated environment (December 2025).

## Core Dependencies
These are essential for the core functionality of the Spandrel project (Cosmology and DDT Solver).

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | `2.4.0` | Numerical computations, array handling |
| `pandas` | `2.2.3` | Data manipulation (Pantheon+ dataset) |
| `scipy` | `1.14.1` | Integration, interpolation, optimization |
| `matplotlib` | `3.9.4` | Plotting and visualization |

## Performance Dependencies (Pinned for AVX/CUDA)
These enable JIT acceleration and are required for high-performance simulation runs.

| Package | Version | Feature |
|---------|---------|---------|
| `numba` | `0.63.1` | JIT compilation (AVX/AVX2/Parallel) |
| `llvmlite` | `0.44.0` | Numba backend (LLVM interface) |
| `mlx` | `0.22.1` | Apple Silicon GPU acceleration (optional) |

## Development Dependencies
Required for running tests and contributing.

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `8.4.2` | Test runner |
| `pytest-cov` | `6.1.1` | Test coverage reporting |
| `setuptools`| `75.6.0` | Build backend |

## Installation

Install all dependencies from the pinned file:
```bash
pip install -r requirements.txt
```

Install the package in editable mode:
```bash
pip install -e .
```
