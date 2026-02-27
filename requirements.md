# Project Requirements

To ensure a reproducible build, the following dependencies are required. These
versions are pinned based on the current validated environment (February 2026).

## External Sub-module Dependency

| Package | Version / Pin | Purpose |
|---------|---------------|---------|
| `spandrel-core` | `git@52c2c9c` | Shared physical constants and core primitives |

spandrel-core is automatically installed via `pip install -e .` using the
pinned commit SHA in `pyproject.toml`. When the pin is updated, regenerate
`requirements.txt` with `pip freeze > requirements.txt`.

## Core Dependencies

These are essential for the core functionality of the Spandrel project
(cosmology analysis and DDT solver).

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | `2.3.5` | Numerical computations, array handling |
| `pandas` | `2.3.1` | Data manipulation (Pantheon+ dataset) |
| `scipy` | `1.16.3` | Integration, interpolation, optimization |
| `matplotlib` | `3.10.8` | Plotting and visualization |

## Performance Dependencies (Pinned for AVX/CUDA)

These enable JIT acceleration and are required for high-performance simulation
runs.

| Package | Version | Feature |
|---------|---------|---------|
| `numba` | `0.63.1` | JIT compilation (AVX/AVX2/Parallel) |
| `llvmlite` | `0.46.0` | Numba backend (LLVM interface) |
| `mlx` | `0.22.1` | Apple Silicon GPU acceleration (macOS only, optional) |

## Development Dependencies

Required for running tests and contributing.

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `8.4.2` | Test runner |
| `pytest-cov` | `6.1.1` | Test coverage reporting |
| `setuptools` | `80.9.0` | Build backend |

## Installation

Install all dependencies from the pinned file:
```bash
pip install -r requirements.txt
```

Install the package in editable mode (includes spandrel-core from git):
```bash
pip install -e .
```

Install with optional GPU acceleration (macOS/Apple Silicon only):
```bash
pip install -e ".[gpu]"
```

Install with optional JIT acceleration (all platforms):
```bash
pip install -e ".[jit]"
```
