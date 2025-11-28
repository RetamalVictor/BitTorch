# Packed Ternary Weight Format

This document specifies the binary format for storing ternary weights in BitTorch's inference modules.

## Overview

Ternary weights take values from the set {-1, 0, +1}. For inference, we pack these into a compact 2-bit representation to achieve ~16x memory reduction compared to FP32 storage.

## Encoding

Each ternary value is encoded using 2 bits:

| Value | Binary |
|-------|--------|
| 0     | `00`   |
| +1    | `01`   |
| -1    | `10`   |
| (reserved) | `11` |

The `11` encoding is reserved for future use (e.g., special values, sparsity markers).

## Packing Layout

### Byte Layout

Four ternary weights are packed into each byte (2 bits × 4 = 8 bits):

```
Byte: [w3 w3 | w2 w2 | w1 w1 | w0 w0]
       ^^^^^   ^^^^^   ^^^^^   ^^^^^
       bits    bits    bits    bits
       6-7     4-5     2-3     0-1
```

Weight order within a byte (LSB to MSB):
- Bits 0-1: weight index 0
- Bits 2-3: weight index 1
- Bits 4-5: weight index 2
- Bits 6-7: weight index 3

### Matrix Layout

For a weight matrix of shape `(out_features, in_features)`:

1. **Row-major order**: Weights are packed row by row (each row = one output channel)
2. **Contiguous packing**: 4 consecutive weights along `in_features` dimension are packed into one byte
3. **Padding**: `in_features` is padded to a multiple of 4 if necessary (padded values = 0)

```
Original shape: (out_features, in_features)
Packed shape:   (out_features, ceil(in_features / 4))  [dtype: uint8]
```

### Example

For a (2, 6) weight matrix:
```
Original weights:
  Row 0: [+1, -1,  0, +1, -1,  0]
  Row 1: [ 0, +1, +1, -1,  0, -1]

Binary encoding:
  Row 0: [01, 10, 00, 01, 10, 00, 00, 00]  (padded to 8 values)
  Row 1: [00, 01, 01, 10, 00, 10, 00, 00]  (padded to 8 values)

Packed bytes:
  Row 0: [0b01_00_10_01, 0b00_00_00_10] = [0x49, 0x02]
  Row 1: [0b10_01_01_00, 0b00_00_10_00] = [0x64, 0x08]

Packed tensor shape: (2, 2) uint8
```

## Scale Storage

Per-channel scales are stored as a separate tensor:

- **Shape**: `(out_features,)`
- **Dtype**: `float16` or `float32`
- **Purpose**: Effective weight = ternary_value × scale

The scale is computed during quantization as:
```python
scale[j] = mean(|w[j, :]|)  # or max, depending on config
```

## Bias Storage

Optional bias tensor:

- **Shape**: `(out_features,)`
- **Dtype**: Same as activations (typically `float16` or `float32`)

## Complete Inference Module State

A `TernaryLinearInference` module stores:

| Attribute | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `weight_packed` | `(out_features, ceil(in_features/4))` | `uint8` | Packed ternary weights |
| `scale` | `(out_features,)` | `float16/32` | Per-channel scaling factors |
| `bias` | `(out_features,)` | `float16/32` | Optional bias (may be None) |
| `in_features` | scalar | `int` | Original input dimension |
| `out_features` | scalar | `int` | Output dimension |

## Memory Comparison

| Format | Bytes per weight | 4096×4096 matrix |
|--------|------------------|------------------|
| FP32   | 4.0              | 64 MB            |
| FP16   | 2.0              | 32 MB            |
| INT8   | 1.0              | 16 MB            |
| Ternary packed | 0.25     | 4 MB             |

Plus scales: `out_features × 2 bytes` (FP16) = 8 KB for 4096 outputs.

**Total for 4096×4096**: ~4.008 MB (vs 64 MB FP32 = **16x reduction**)

## Alignment Considerations

For optimal memory access patterns:

1. **Rows should be 16-byte aligned** when possible (for vectorized loads)
2. **Padding to multiples of 16** (64 weights) is recommended for CUDA kernels
3. Minimal implementation: pad to multiple of 4 (required for byte packing)

## Implementation Notes

### Packing (Python)

```python
def pack_ternary(w_tern: Tensor) -> Tensor:
    """Pack ternary weights {-1, 0, +1} into uint8.

    Args:
        w_tern: Ternary weights, shape (out_features, in_features)
                Values must be in {-1, 0, +1}

    Returns:
        Packed weights, shape (out_features, ceil(in_features/4)), dtype uint8
    """
    # Encode: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10
    encoded = torch.where(w_tern == 1, 1, torch.where(w_tern == -1, 2, 0))
    # Pad to multiple of 4
    # Pack 4 values per byte
    ...
```

### Unpacking (Python)

```python
def unpack_ternary(packed: Tensor, in_features: int) -> Tensor:
    """Unpack uint8 to ternary weights {-1, 0, +1}.

    Args:
        packed: Packed weights, shape (out_features, packed_in_features)
        in_features: Original input dimension (for removing padding)

    Returns:
        Ternary weights, shape (out_features, in_features), dtype int8
    """
    # Extract 2-bit values
    # Decode: 0b00 -> 0, 0b01 -> +1, 0b10 -> -1
    ...
```

## Version History

- **v0.2.0**: Initial packed ternary format specification
