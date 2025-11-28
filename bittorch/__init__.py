"""BitTorch: High-performance low-precision backend for PyTorch."""

__version__ = "0.1.1"

# Import torch first to load shared libraries needed by the C++ extension
import torch  # noqa: F401

# Import submodules
from . import ops
from . import nn
from . import quant
from . import utils

# Try to import the C++ extension
try:
    from . import _C
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


def has_cuda_ext() -> bool:
    """Check if the CUDA extension is available."""
    return _HAS_CUDA_EXT
