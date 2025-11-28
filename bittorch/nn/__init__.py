"""Neural network modules with low-precision weights."""

from .ternary_linear import TernaryLinear
from .ternary_linear_cuda import TernaryLinearCUDA

__all__ = ["TernaryLinear", "TernaryLinearCUDA"]
