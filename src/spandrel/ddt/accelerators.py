"""
Accelerator definitions for Spandrel DDT solver.

Handles conditional importing of Numba and CUDA support.
If Numba is present, provides JIT compilation decorators.
If absent, provides no-op decorators that return the original Python/NumPy function.

Target Architectures:
    - CPU: AVX/AVX2 via Numba LLVM
    - GPU: CUDA 12 (SM89) via Numba CUDA
"""

import warnings
import os

# Configuration
ENABLE_JIT = os.getenv("SPANDREL_JIT", "1") == "1"
ENABLE_CUDA = os.getenv("SPANDREL_CUDA", "0") == "1"  # Default off unless requested

# -----------------------------------------------------------------------------
# NUMBA JIT (CPU)
# -----------------------------------------------------------------------------
try:
    if not ENABLE_JIT:
        raise ImportError("JIT disabled by configuration")

    from numba import jit, njit, prange, float64, int32, void
    
    # Standard CPU JIT configuration
    # fastmath=True allows reordering of FP operations (SIMD friendly)
    # cache=True speeds up subsequent launches
    JIT_CONFIG = {
        'nopython': True,
        'fastmath': True,
        'cache': True,
        'parallel': True
    }

    def cpu_jit(func):
        """Decorator for CPU JIT compilation."""
        return njit(**JIT_CONFIG)(func)

    HAVE_NUMBA = True

except ImportError:
    # Fallback: No-op decorator
    def cpu_jit(func):
        """No-op decorator (standard Python execution)."""
        return func
        
    def prange(*args):
        """Fallback for parallel range."""
        return range(*args)

    HAVE_NUMBA = False

# -----------------------------------------------------------------------------
# CUDA JIT (GPU)
# -----------------------------------------------------------------------------
try:
    if not ENABLE_CUDA:
        raise ImportError("CUDA disabled by configuration")

    from numba import cuda
    
    if not cuda.is_available():
        raise ImportError("CUDA driver not detected")

    def gpu_jit(func):
        """Decorator for CUDA JIT compilation."""
        return cuda.jit(func)

    HAVE_CUDA = True

except ImportError:
    # Fallback: No-op decorator
    def gpu_jit(func):
        """No-op decorator (GPU not available)."""
        return func

    HAVE_CUDA = False

# -----------------------------------------------------------------------------
# STATUS REPORT
# -----------------------------------------------------------------------------
def get_acceleration_status() -> dict:
    return {
        "numba_cpu": HAVE_NUMBA,
        "numba_cuda": HAVE_CUDA,
        "mode": "CUDA" if HAVE_CUDA else ("AVX/JIT" if HAVE_NUMBA else "NumPy (Fallback)")
    }
