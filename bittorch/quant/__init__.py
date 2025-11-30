"""Quantization utilities and configurations.

This module provides a unified interface for different quantization schemes:
- Ternary (1.58-bit): {-1, 0, +1} weights (implemented)
- INT4 (4-bit): {-8, ..., +7} weights (placeholder)

Usage:
    # Functional API (recommended for most users)
    from bittorch.quant import ternary_quantize, ternary_quantize_ste

    # Class-based API (for custom quantization pipelines)
    from bittorch.quant import TernaryQuantConfig, TernaryQuantizer
"""

# Base classes for extending quantization
from .base import (
    QuantConfig,
    QuantType,
    Quantizer,
    ScaleType,
    compute_scale,
)

# Ternary quantization (implemented)
from .ternary import (
    TernaryQuantConfig,
    TernaryQuantizer,
    dequantize_ternary,
    ternary_quantize,
    ternary_quantize_ste,
)

# Packed ternary utilities (for inference)
from .ternary_packed import (
    pack_ternary,
    unpack_ternary,
    pack_ternary_with_scale,
    pack_ternary_transposed,
    unpack_ternary_transposed,
    get_packed_size,
    get_memory_reduction,
)

# INT4 quantization (placeholder for future)
from .int4 import (
    Int4QuantConfig,
    Int4Quantizer,
    dequantize_int4,
    int4_quantize,
    int4_quantize_ste,
)

__all__ = [
    # Base
    "QuantConfig",
    "QuantType",
    "Quantizer",
    "ScaleType",
    "compute_scale",
    # Ternary
    "TernaryQuantConfig",
    "TernaryQuantizer",
    "ternary_quantize",
    "ternary_quantize_ste",
    "dequantize_ternary",
    # Packed ternary (inference)
    "pack_ternary",
    "unpack_ternary",
    "pack_ternary_with_scale",
    "pack_ternary_transposed",
    "unpack_ternary_transposed",
    "get_packed_size",
    "get_memory_reduction",
    # INT4 (placeholder)
    "Int4QuantConfig",
    "Int4Quantizer",
    "int4_quantize",
    "int4_quantize_ste",
    "dequantize_int4",
]
