# Spandrel -- Type Ia supernova DDT simulation and cosmological analysis
#
# WHY: Reproducible compute environment that pins all native and Python
#      dependencies so results can be reproduced independently of the host OS.
# WHAT: Python 3.12 slim image with all spandrel extras pre-installed.
# HOW:  docker build -t spandrel .
#       docker run --rm -v $(pwd)/results:/app/results spandrel spandrel --help

FROM python:3.12-slim

LABEL maintainer="Eirikr"
LABEL description="Spandrel: Type Ia supernovae DDT simulation and cosmological hypothesis testing"
LABEL license="GPL-2.0-only"

# System dependencies
#   - gcc / g++: Numba LLVM compilation
#   - libgomp1: OpenMP runtime for numba parallel loops
#   - libhdf5-dev: optional HDF5 support via h5py
# Non-interactive frontend suppresses apt prompts in CI.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files needed for dependency installation first so Docker can
# cache this layer independently of source changes.
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

# Install spandrel with all CPU extras.
# MLX is macOS-only so we exclude it; the platform marker in pyproject.toml
# already guards this, but we make it explicit here for clarity.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[cpu,jit]"

# Copy the Pantheon+SH0ES data file (optional -- mount at runtime instead).
COPY data/ ./data/

# Copy the test suite for optional verification.
COPY tests/ ./tests/

# Non-root user for security.
RUN useradd --create-home --shell /bin/bash spandrel
USER spandrel

# Default entrypoint delegates to the installed CLI.
ENTRYPOINT ["spandrel"]
CMD ["--help"]
